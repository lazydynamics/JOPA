"""Observations: raw datum → Gaussian message on a block's state slice.

Frozen sensors (`Frozen`, `VAEObservation`, `PoseMotionObservation`) only
encode; `LearnedVAE` also refines its weights in the M-step.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import (
    CALIBRATION_COVERAGE,
    CALIBRATION_OFFSET_CLIP,
    DEFAULT_IMG_SIZE,
    DEFAULT_N_FRAMES,
    DEFAULT_SEED,
    EPS_DIVISION,
    GOAL_EIGENVALUE_FLOOR,
    GOAL_MOTION_PRECISION,
    GOAL_POSE_MEAN_PRECISION,
    GOAL_PRECISION_RIDGE,
    GOAL_SALIENCY_FLOOR,
    GOAL_SALIENCY_MAX_FLOOR,
    LOG_STD_CLIP,
    PROB_CLIP,
    VAE_M_STEP_BETA_RECON,
    VAE_M_STEP_COUNT,
    VAE_M_STEP_LR,
)
from .distributions import Gaussian
from .nn.vae import _as_batch, _latest_frame


class Observation:
    """Maps a raw datum to a Gaussian message on its block's state slice.

    Subclasses define `message(datum) -> Gaussian`. Learnable observations
    additionally define `update(data, post_means, post_covs)` for the M-step.
    """
    learnable = False


class Frozen(Observation):
    """Fixed encoder: wraps `encode: datum -> Gaussian` and optional `decode: z -> datum`."""

    def __init__(self, encode: Callable, decode: Callable | None = None):
        self.encode = encode
        if decode is not None:
            self.decode = decode

    def message(self, datum) -> Gaussian:
        return self.encode(datum)


def _kl_diag_vs_full(mu1, log_std1, mu2, cov2):
    """KL( N(mu1, diag(σ1²)) || N(mu2, Σ2) )."""
    d = mu1.shape[0]
    var1 = jnp.exp(2.0 * log_std1)
    prec2 = jnp.linalg.inv(cov2)
    diff = mu2 - mu1
    return 0.5 * (jnp.sum(jnp.diag(prec2) * var1) + diff @ prec2 @ diff
                  - d + jnp.linalg.slogdet(cov2)[1] - 2.0 * jnp.sum(log_std1))


def _vae_m_step_loss(params, model, images, post_means, post_covs, z_rng, beta_recon):
    """ELBO for the VAE M-step: BCE reconstruction + KL(encoder ‖ smoothed posterior).

    `images` is the encoder input — a window `(B, K, 28, 28)` (multi-frame)
    or `(B, 28, 28)` (single-frame). The decoder reconstructs the full input —
    the whole K-frame window for multi-frame, so the latent must encode motion."""
    mu_enc, log_std_enc = model.apply(params, images, method=model.encode)
    log_std_enc = jnp.clip(log_std_enc, *LOG_STD_CLIP)
    z = mu_enc + jnp.exp(log_std_enc) * jax.random.normal(z_rng, mu_enc.shape)
    recon = jnp.clip(model.apply(params, z, method=model.decode), *PROB_CLIP)
    flat = images.reshape(images.shape[0], -1)
    bce = -jnp.sum(flat * jnp.log(recon) + (1 - flat) * jnp.log(1 - recon), axis=-1)
    kl = jax.vmap(_kl_diag_vs_full)(mu_enc, log_std_enc, post_means, post_covs)
    return jnp.mean(beta_recon * bce + kl)


class LearnedVAE(Observation):
    """VAE observation whose encoder produces messages and whose weights are
    refined in the M-step. Follows the VAE's `n_frames` setting — when the
    underlying VAE is multi-frame, the encoder input is a K-frame window
    `(K, 28, 28)` and the decoder reconstructs the whole window; `decode`
    returns its latest frame."""
    learnable = True

    def __init__(self, model, params, lr=VAE_M_STEP_LR,
                 n_m_steps=VAE_M_STEP_COUNT,
                 beta_recon=VAE_M_STEP_BETA_RECON, seed=DEFAULT_SEED):
        self.model = model
        self.params = params
        self.n_frames = getattr(model, "n_frames", DEFAULT_N_FRAMES)
        self.img_size = getattr(model, "img_size", DEFAULT_IMG_SIZE)
        self.n_m_steps = n_m_steps
        self.beta_recon = beta_recon
        self._tx = optax.adam(lr)
        self._opt_state = self._tx.init(params)
        self._rng = jax.random.PRNGKey(seed)
        self.loss_history: list[float] = []

    def _shape(self, x):
        return _as_batch(x, self.n_frames, self.img_size)

    def message(self, image) -> Gaussian:
        mu, log_std = self.model.apply(self.params, self._shape(image), method=self.model.encode)
        log_std = jnp.clip(log_std[0], *LOG_STD_CLIP)
        lam = jnp.diag(1.0 / jnp.exp(2.0 * log_std))
        return Gaussian(eta=lam @ mu[0], lam=lam)

    def update(self, data, post_means, post_covs):
        images = jnp.stack([jnp.asarray(im) for im in data])

        @jax.jit
        def step(params, opt_state, z_rng):
            loss, grads = jax.value_and_grad(_vae_m_step_loss)(
                params, self.model, images, post_means, post_covs, z_rng, self.beta_recon)
            updates, opt_state = self._tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        for _ in range(self.n_m_steps):
            self._rng, z_rng = jax.random.split(self._rng)
            self.params, self._opt_state, loss = step(self.params, self._opt_state, z_rng)
            self.loss_history.append(float(loss))

    def decode(self, z):
        return _latest_frame(self.model.apply(self.params, jnp.asarray(z).reshape(1, -1),
                                              method=self.model.decode), self.img_size)


class VAEObservation(Observation):
    """Frozen probabilistic VAE sensor: pixels become Gaussian messages.

    Unlike a deterministic feature wrapper, this preserves the encoder's
    heteroscedastic covariance for filtering and uncertainty telemetry. The
    weights never change once constructed.
    """
    learnable = False

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.n_frames = int(getattr(model, "n_frames", DEFAULT_N_FRAMES))
        self.img_size = int(getattr(model, "img_size", DEFAULT_IMG_SIZE))
        self.dim = int(model.latent_dim)
        self._encode_apply = jax.jit(
            lambda p, x: self.model.apply(p, x, method=self.model.encode))

    def encode_batch(self, observations):
        x = jnp.asarray(observations)
        expected = ((self.img_size, self.img_size) if self.n_frames == 1
                    else (self.n_frames, self.img_size, self.img_size))
        if x.ndim != len(expected) + 1 or x.shape[1:] != expected:
            raise ValueError(f"expected batch (B,{','.join(map(str, expected))}), got {x.shape}")
        mu, log_std = self._encode_apply(self.params, x)
        return mu, jnp.clip(log_std, *LOG_STD_CLIP)

    def encode(self, observation):
        x = _as_batch(observation, self.n_frames, self.img_size)
        mu, log_std = self.encode_batch(x)
        return mu[0], log_std[0]

    def message(self, observation) -> Gaussian:
        mu, log_std = self.encode(observation)
        lam = jnp.diag(jnp.exp(-2.0 * log_std))
        return Gaussian(eta=lam @ mu, lam=lam)

    def decode(self, z):
        reconstruction = self.model.apply(
            self.params, jnp.asarray(z).reshape(1, -1),
            method=self.model.decode)
        return _latest_frame(reconstruction, self.img_size)


class PoseMotionObservation(Observation):
    """Frozen six-dimensional pose/motion Gaussian observation.

    A frame window emits the model's heteroscedastic pose/motion posterior.
    A single frame is treated as a static image goal and emits
    [pose(frame), 0, 0]. Per-dimension log-standard-deviation offsets are
    calibrated from held-out pixel/action innovations and never alter means.
    """

    learnable = False

    def __init__(self, model, params, log_std_offsets=None):
        required = ("pose_dim", "motion_dim", "n_frames", "img_size")
        if any(not hasattr(model, name) for name in required):
            raise TypeError("PoseMotionObservation requires a PoseMotionVAE")
        self.model = model
        self.params = params
        self.pose_dim = int(model.pose_dim)
        self.motion_dim = int(model.motion_dim)
        self.dim = self.pose_dim + self.motion_dim
        self.n_frames = int(model.n_frames)
        self.img_size = int(model.img_size)
        offsets = (
            jnp.zeros(self.dim, dtype=jnp.float32)
            if log_std_offsets is None
            else jnp.asarray(log_std_offsets, dtype=jnp.float32))
        if offsets.shape != (self.dim,):
            raise ValueError(
                f"log_std_offsets must have shape ({self.dim},), "
                f"got {offsets.shape}")
        if not bool(jnp.all(jnp.isfinite(offsets))):
            raise ValueError("log_std_offsets must be finite")
        self.log_std_offsets = offsets
        self._encode_apply = jax.jit(
            lambda p, x: self.model.apply(
                p, x, method=self.model.encode))
        self._pose_apply = jax.jit(
            lambda p, x: self.model.apply(
                p, x, method=self.model.encode_static))

    def set_log_std_offsets(self, offsets):
        offsets = jnp.asarray(offsets, dtype=jnp.float32)
        if offsets.shape != (self.dim,):
            raise ValueError(
                f"calibration offsets must have shape ({self.dim},)")
        if not bool(jnp.all(jnp.isfinite(offsets))):
            raise ValueError("calibration offsets must be finite")
        self.log_std_offsets = offsets
        return self

    def _calibrated(self, log_std):
        return jnp.clip(
            log_std + self.log_std_offsets, *LOG_STD_CLIP)

    def encode_batch(self, windows):
        x = jnp.asarray(windows)
        expected = (
            self.n_frames, self.img_size, self.img_size)
        if x.ndim != 4 or x.shape[1:] != expected:
            raise ValueError(
                f"expected batch (B,{self.n_frames},"
                f"{self.img_size},{self.img_size}), got {x.shape}")
        mean, log_std = self._encode_apply(self.params, x)
        return mean, self._calibrated(log_std)

    def encode_goal_batch(self, frames):
        x = jnp.asarray(frames)
        expected = (self.img_size, self.img_size)
        if x.ndim != 3 or x.shape[1:] != expected:
            raise ValueError(
                f"expected goal batch (B,{self.img_size},"
                f"{self.img_size}), got {x.shape}")
        mean, log_std = self._pose_apply(self.params, x)
        # encode_static guarantees the last motion_dim means are exact zeros.
        return mean, self._calibrated(log_std)

    def encode(self, observation):
        x = jnp.asarray(observation)
        if x.shape == (self.img_size, self.img_size):
            mean, log_std = self.encode_goal_batch(x[None])
        elif x.shape == (
                self.n_frames, self.img_size, self.img_size):
            mean, log_std = self.encode_batch(x[None])
        else:
            raise ValueError(
                "expected a static goal frame or a chronological "
                f"{self.n_frames}-frame window, got {x.shape}")
        return mean[0], log_std[0]

    def message(self, observation) -> Gaussian:
        mean, log_std = self.encode(observation)
        precision = jnp.diag(jnp.exp(-2.0 * log_std))
        return Gaussian(eta=precision @ mean, lam=precision)

    @staticmethod
    def calibration_offsets(
            innovations, base_log_stds, coverage=CALIBRATION_COVERAGE,
            clip=CALIBRATION_OFFSET_CLIP):
        """Quantile-match per-dimension innovation scales.

        Inputs are latent innovations and encoder log standard deviations
        derived from calibration pixels/actions. No physical state is accepted.
        """
        from statistics import NormalDist

        innovation = np.asarray(innovations, dtype=np.float64)
        log_std = np.asarray(base_log_stds, dtype=np.float64)
        if innovation.ndim != 2 or innovation.shape != log_std.shape:
            raise ValueError(
                "innovations and base_log_stds must have equal (N,D) shape")
        if innovation.shape[0] < 2:
            raise ValueError("at least two calibration innovations are required")
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie strictly between zero and one")
        if not (
                np.all(np.isfinite(innovation))
                and np.all(np.isfinite(log_std))):
            raise ValueError("calibration inputs must be finite")
        z_value = NormalDist().inv_cdf(0.5 + coverage / 2.0)
        standardized = np.abs(innovation) / np.maximum(
            np.exp(log_std), EPS_DIVISION)
        scale = np.quantile(standardized, coverage, axis=0)
        offsets = np.log(np.maximum(scale / z_value, EPS_DIVISION))
        return np.clip(offsets, clip[0], clip[1]).astype(np.float32)

    def calibrate(self, innovations, base_log_stds,
                  coverage=CALIBRATION_COVERAGE):
        offsets = self.calibration_offsets(
            innovations, base_log_stds, coverage=coverage)
        return self.set_log_std_offsets(offsets)

    def decode(self, state):
        reconstruction = self.model.apply(
            self.params, jnp.asarray(state).reshape(1, -1),
            method=self.model.decode)
        return _latest_frame(reconstruction, self.img_size)

    def goal_precision(
            self, goal_frame, background=None,
            pose_mean_precision=GOAL_POSE_MEAN_PRECISION,
            motion_precision=GOAL_MOTION_PRECISION,
            eigenvalue_floor=GOAL_EIGENVALUE_FLOOR):
        """Decoder-Jacobian image metric for a static image goal."""
        if pose_mean_precision <= 0 or motion_precision <= 0:
            raise ValueError("goal precision scales must be positive")
        frame = jnp.asarray(goal_frame, dtype=jnp.float32)
        if frame.shape != (self.img_size, self.img_size):
            raise ValueError(
                f"expected goal frame {(self.img_size, self.img_size)}, "
                f"got {frame.shape}")
        goal_state, _ = self.encode(frame)
        pose = goal_state[:self.pose_dim]
        if background is None:
            background_array = jnp.zeros_like(frame)
        else:
            background_array = jnp.asarray(
                background, dtype=jnp.float32)
            if background_array.shape != frame.shape:
                raise ValueError("background must match the goal frame")

        def render_pose(value):
            decoded = self.model.apply(
                self.params, value.reshape(1, -1),
                method=self.model.decode)
            return decoded.reshape(self.img_size, self.img_size)

        jacobian = jax.jacrev(render_pose)(pose)
        saliency = jnp.abs(frame - background_array)
        saliency = saliency / jnp.maximum(
            jnp.max(saliency), GOAL_SALIENCY_MAX_FLOOR)
        weight = GOAL_SALIENCY_FLOOR + saliency
        flat_jacobian = jacobian.reshape((-1, self.pose_dim))
        pose_precision = (
            flat_jacobian.T
            @ (weight.reshape(-1, 1) * flat_jacobian))
        pose_precision = 0.5 * (
            pose_precision + pose_precision.T)
        pose_precision = (
            pose_precision + GOAL_PRECISION_RIDGE * jnp.eye(self.pose_dim))
        diagonal_mean = jnp.mean(jnp.diag(pose_precision))
        pose_precision = (
            pose_precision
            * (pose_mean_precision
               / jnp.maximum(diagonal_mean, EPS_DIVISION)))
        if eigenvalue_floor > 0.0:
            # The decoder Jacobian is close to rank-deficient (the saliency
            # weighting makes this an edge metric), so normalising the mean
            # diagonal alone leaves latent directions charged orders of
            # magnitude below nominal. Physically distinct arm configurations
            # then sit inside the goal's own credible ball and the planner has
            # no restoring force along those directions. Floor the spectrum,
            # then restore the nominal mean diagonal.
            values, vectors = jnp.linalg.eigh(pose_precision)
            values = jnp.maximum(
                values, eigenvalue_floor * pose_mean_precision)
            pose_precision = (vectors * values) @ vectors.T
            pose_precision = 0.5 * (pose_precision + pose_precision.T)
            diagonal_mean = jnp.mean(jnp.diag(pose_precision))
            pose_precision = (
                pose_precision
                * (pose_mean_precision
                   / jnp.maximum(diagonal_mean, EPS_DIVISION)))
        full = jnp.zeros((self.dim, self.dim), dtype=jnp.float32)
        full = full.at[:self.pose_dim, :self.pose_dim].set(
            pose_precision)
        full = full.at[self.pose_dim:, self.pose_dim:].set(
            motion_precision * jnp.eye(self.motion_dim))
        return full


def _as_observation(obj) -> Observation:
    return obj if isinstance(obj, Observation) else Frozen(obj)
