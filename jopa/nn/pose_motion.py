"""Structured pixel-only sensor: pose from one frame, motion from pose deltas."""
from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from ..config import (
    BACKGROUND_SAMPLE_FRAMES,
    BOOTSTRAP_A_DECAY,
    DEFAULT_MOTION_DIM,
    DEFAULT_POSE_DIM,
    DEFAULT_WINDOW_FRAMES,
    EPS_DIVISION,
    EPS_LATENT_SCALE,
    EPS_NORMALIZER,
    EPS_SQRT,
    IMG_SIZE_MEDIUM,
    LOG_STD_BIAS_INIT,
    LOG_STD_CLIP,
    MOTION_HIDDEN_WIDTH,
    PHASE_TWO_DECORRELATION_SCALE,
    PIXEL_SENSOR_CHANNELS,
    SUPPORTED_IMG_SIZES,
    TRAINING_HISTORY_POINTS,
    PoseMotionTrainingConfig,
)
from .layers import _Decoder, _Encoder
from .losses import decorrelation


class _MotionHead(nn.Module):
    """Temporal Gaussian head operating only on consecutive pose encodings."""

    pose_dim: int = DEFAULT_POSE_DIM
    motion_dim: int = DEFAULT_MOTION_DIM
    n_frames: int = DEFAULT_WINDOW_FRAMES
    hidden_dim: int = MOTION_HIDDEN_WIDTH

    @nn.compact
    def __call__(self, pose_sequence):
        if pose_sequence.ndim != 3:
            raise ValueError("pose_sequence must have shape (B,K,pose_dim)")
        # Bias-free mean layers make a repeated-frame window exactly
        # zero-motion by construction; training adds an explicit rest penalty.
        delta = pose_sequence[:, 1:] - pose_sequence[:, :-1]
        x = delta.reshape((delta.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_dim, use_bias=False)(x))
        x = nn.relu(nn.Dense(self.hidden_dim, use_bias=False)(x))
        mu = nn.Dense(self.motion_dim, use_bias=False)(x)
        log_std = nn.Dense(
            self.motion_dim,
            bias_init=nn.initializers.constant(LOG_STD_BIAS_INIT))(x)
        return mu, log_std


class PoseMotionVAE(nn.Module):
    """Pixel-only structured Gaussian sensor.

    The current frame alone determines pose. A temporal head sees four
    consecutive pose encodings and determines motion. The decoder accepts
    pose only, so the six-dimensional state has the fixed interpretation
    [pose(4), motion(2)] without kinematic or simulator supervision.
    """

    pose_dim: int = DEFAULT_POSE_DIM
    motion_dim: int = DEFAULT_MOTION_DIM
    ch: int = PIXEL_SENSOR_CHANNELS
    n_frames: int = DEFAULT_WINDOW_FRAMES
    img_size: int = IMG_SIZE_MEDIUM
    motion_hidden_dim: int = MOTION_HIDDEN_WIDTH

    def setup(self):
        if self.img_size not in SUPPORTED_IMG_SIZES:
            raise ValueError(
                f"img_size must be 28, 64 or 128, got {self.img_size}")
        if self.n_frames < 2:
            raise ValueError("PoseMotionVAE requires at least two frames")
        if self.pose_dim < 1 or self.motion_dim < 1:
            raise ValueError("pose_dim and motion_dim must be positive")
        self.pose_encoder = _Encoder(
            self.pose_dim, self.ch, 1, self.img_size)
        self.pose_decoder = _Decoder(
            self.ch, 1, self.img_size)
        self.motion_head = _MotionHead(
            self.pose_dim, self.motion_dim, self.n_frames,
            self.motion_hidden_dim)

    @property
    def latent_dim(self):
        return self.pose_dim + self.motion_dim

    def architecture(self):
        """JSON-serializable values that define checkpoint compatibility."""
        return {
            "kind": "PoseMotionVAE",
            "pose_dim": int(self.pose_dim),
            "motion_dim": int(self.motion_dim),
            "latent_dim": int(self.latent_dim),
            "channels": int(self.ch),
            "n_frames": int(self.n_frames),
            "img_size": int(self.img_size),
            "motion_hidden_dim": int(self.motion_hidden_dim),
        }

    def encode_pose(self, frames):
        x = jnp.asarray(frames)
        if x.ndim != 3:
            raise ValueError("encode_pose expects frames with shape (B,S,S)")
        return self.pose_encoder(x)

    def encode_motion(self, pose_sequence):
        return self.motion_head(pose_sequence)

    def encode(self, windows):
        x = jnp.asarray(windows)
        if x.ndim != 4 or x.shape[1] != self.n_frames:
            raise ValueError(
                f"encode expects (B,{self.n_frames},S,S), got {x.shape}")
        batch = x.shape[0]
        pose_mu, pose_log_std = self.pose_encoder(
            x.reshape((-1, self.img_size, self.img_size)))
        pose_mu = pose_mu.reshape((batch, self.n_frames, self.pose_dim))
        pose_log_std = pose_log_std.reshape(
            (batch, self.n_frames, self.pose_dim))
        motion_mu, motion_log_std = self.motion_head(pose_mu)
        return (
            jnp.concatenate([pose_mu[:, -1], motion_mu], axis=-1),
            jnp.concatenate([pose_log_std[:, -1], motion_log_std], axis=-1),
        )

    def encode_static(self, frames):
        """Encode a goal frame as pose with an exact zero-motion mean."""
        pose_mu, pose_log_std = self.encode_pose(frames)
        repeated = jnp.repeat(
            pose_mu[:, None, :], self.n_frames, axis=1)
        _, motion_log_std = self.motion_head(repeated)
        zero = jnp.zeros((pose_mu.shape[0], self.motion_dim), pose_mu.dtype)
        return (
            jnp.concatenate([pose_mu, zero], axis=-1),
            jnp.concatenate([pose_log_std, motion_log_std], axis=-1),
        )

    def decode(self, state):
        z = jnp.asarray(state)
        return self.pose_decoder(z[..., :self.pose_dim])

    def __call__(self, windows, z_rng):
        mu, log_std = self.encode(windows)
        state = mu + jnp.exp(log_std) * jax.random.normal(z_rng, mu.shape)
        return self.decode(state), mu, log_std


def train_pose_motion_vae(
    frame_seqs,
    control_seqs,
    *,
    config: PoseMotionTrainingConfig | None = None,
    pose_dim: int = PoseMotionTrainingConfig.pose_dim,
    motion_dim: int = PoseMotionTrainingConfig.motion_dim,
    n_frames: int = PoseMotionTrainingConfig.n_frames,
    rollout: int = PoseMotionTrainingConfig.rollout,
    ch: int = PoseMotionTrainingConfig.ch,
    motion_hidden_dim: int = PoseMotionTrainingConfig.motion_hidden_dim,
    img_size: int = PoseMotionTrainingConfig.img_size,
    pose_steps: int = PoseMotionTrainingConfig.pose_steps,
    controlled_steps: int = PoseMotionTrainingConfig.controlled_steps,
    batch_size: int = PoseMotionTrainingConfig.batch_size,
    lr: float = PoseMotionTrainingConfig.lr,
    lr_decay_alpha: float = PoseMotionTrainingConfig.lr_decay_alpha,
    reconstruction_weight: float = PoseMotionTrainingConfig.reconstruction_weight,
    foreground_focus: float = PoseMotionTrainingConfig.foreground_focus,
    variance_weight: float = PoseMotionTrainingConfig.variance_weight,
    covariance_weight: float = PoseMotionTrainingConfig.covariance_weight,
    visual_distance_weight: float = PoseMotionTrainingConfig.visual_distance_weight,
    prediction_weight: float = PoseMotionTrainingConfig.prediction_weight,
    prediction_nll_weight: float = PoseMotionTrainingConfig.prediction_nll_weight,
    inverse_weight: float = PoseMotionTrainingConfig.inverse_weight,
    fixed_point_weight: float = PoseMotionTrainingConfig.fixed_point_weight,
    zero_motion_weight: float = PoseMotionTrainingConfig.zero_motion_weight,
    kl_weight: float = PoseMotionTrainingConfig.kl_weight,
    beta_nll: float = PoseMotionTrainingConfig.beta_nll,
    controlled_ramp: int = PoseMotionTrainingConfig.controlled_ramp,
    seed: int = PoseMotionTrainingConfig.seed,
    verbose: bool = True,
    callback: Callable | None = None,
):
    """Train the structured state from agent-generated pixels and actions.

    Phase one trains only pose from individual images with foreground
    reconstruction, anti-collapse, covariance decorrelation, and preservation
    of foreground visual distances. Phase two retains those losses and adds a
    disposable recursive action-conditioned predictor, beta-NLL, inverse
    action prediction, repeated-frame zero motion, and a zero-action fixed
    point. The predictor is returned only as diagnostics; deployed dynamics
    remain JOPA's conjugate q(A,B,W).

    Hyperparameters are documented as the fields of
    :class:`jopa.config.PoseMotionTrainingConfig`; passing a preset as
    ``config`` supersedes the individual keywords.
    """
    if config is not None:
        return train_pose_motion_vae(
            frame_seqs, control_seqs, verbose=verbose, callback=callback,
            **config.kwargs())
    if n_frames < 2:
        raise ValueError("n_frames must be at least two")
    if rollout < 1:
        raise ValueError("rollout must be positive")
    if pose_steps < 0 or controlled_steps < 0:
        raise ValueError("pose_steps and controlled_steps must be non-negative")
    if pose_steps + controlled_steps == 0:
        raise ValueError("at least one training step is required")
    if len(frame_seqs) != len(control_seqs):
        raise ValueError("frame_seqs and control_seqs must have equal length")
    if not frame_seqs:
        raise ValueError("at least one trajectory is required")

    frames = [np.asarray(sequence, dtype=np.float32)
              for sequence in frame_seqs]
    controls = [np.asarray(sequence, dtype=np.float32)
                for sequence in control_seqs]
    if controls[0].ndim != 2:
        raise ValueError("controls must have shape (T-1, control_dim)")
    control_dim = controls[0].shape[1]
    pose_refs = []
    controlled_refs = []
    background_samples = []
    span = n_frames + rollout
    for index, (sequence, action) in enumerate(zip(frames, controls)):
        if sequence.ndim != 3 or sequence.shape[1:] != (img_size, img_size):
            raise ValueError(
                f"trajectory {index}: expected frames "
                f"(T,{img_size},{img_size}), got {sequence.shape}")
        expected = (len(sequence) - 1, control_dim)
        if action.shape != expected:
            raise ValueError(
                f"trajectory {index}: expected controls {expected}, "
                f"got {action.shape}")
        pose_refs.extend((index, step) for step in range(len(sequence)))
        controlled_refs.extend(
            (index, start)
            for start in range(max(0, len(sequence) - span + 1)))
        take = np.linspace(
            0, len(sequence) - 1,
            min(BACKGROUND_SAMPLE_FRAMES, len(sequence)), dtype=int)
        background_samples.extend(sequence[take])
    if controlled_steps and not controlled_refs:
        raise ValueError(
            f"controlled training needs {span} consecutive frames")
    background = jnp.asarray(np.median(
        np.stack(background_samples), axis=0).astype(np.float32))

    model = PoseMotionVAE(
        pose_dim=pose_dim, motion_dim=motion_dim, ch=ch,
        n_frames=n_frames, img_size=img_size,
        motion_hidden_dim=motion_hidden_dim)
    latent_dim = pose_dim + motion_dim
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    sensor_params = model.init(
        {"params": init_rng},
        jnp.ones((1, n_frames, img_size, img_size)),
        sample_rng)

    def pose_objective(params, images):
        pose_mu, pose_log_std = model.apply(
            params, images, method=model.encode_pose)
        pose_log_std = jnp.clip(pose_log_std, *LOG_STD_CLIP)
        reconstruction = model.apply(
            params, pose_mu, method=model.decode)
        reconstruction = reconstruction.reshape(images.shape)
        foreground = jax.lax.stop_gradient(jnp.abs(images - background))
        pixel_weight = 1.0 + foreground_focus * foreground
        reconstruction_loss = (
            jnp.sum(pixel_weight * (reconstruction - images) ** 2)
            / jnp.maximum(jnp.sum(pixel_weight), EPS_DIVISION))

        variance_loss, covariance_loss = decorrelation(
            pose_mu, normalize=True)
        paired_images = jnp.roll(images, 1, axis=0)
        paired_pose = jnp.roll(pose_mu, 1, axis=0)
        pair_foreground = jax.lax.stop_gradient(jnp.maximum(
            jnp.abs(images - background),
            jnp.abs(paired_images - background)))
        visual_distance = jnp.sqrt(
            jnp.sum(
                pair_foreground * (images - paired_images) ** 2,
                axis=(1, 2))
            / jnp.maximum(
                jnp.sum(pair_foreground, axis=(1, 2)), EPS_NORMALIZER)
            + EPS_SQRT)
        pose_distance = jnp.sqrt(
            jnp.mean((pose_mu - paired_pose) ** 2, axis=1) + EPS_SQRT)
        visual_distance = visual_distance / jax.lax.stop_gradient(
            jnp.mean(visual_distance) + EPS_NORMALIZER)
        pose_distance = pose_distance / jax.lax.stop_gradient(
            jnp.mean(pose_distance) + EPS_NORMALIZER)
        visual_loss = jnp.mean(
            (visual_distance - pose_distance) ** 2)
        pose_variance = jnp.exp(2.0 * pose_log_std)
        kl = 0.5 * jnp.mean(
            pose_mu ** 2 + pose_variance - 1.0 - 2.0 * pose_log_std)
        total = (
            reconstruction_weight * reconstruction_loss
            + variance_weight * variance_loss
            + covariance_weight * covariance_loss
            + visual_distance_weight * visual_loss
            + kl_weight * kl)
        return total, (
            reconstruction_loss, variance_loss, covariance_loss,
            visual_loss, kl)

    pose_tx = optax.adam(lr)
    pose_opt = pose_tx.init(sensor_params)

    @jax.jit
    def pose_step(params, opt_state, images):
        (loss, aux), gradients = jax.value_and_grad(
            pose_objective, has_aux=True)(params, images)
        updates, opt_state = pose_tx.update(
            gradients, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state, loss, aux)

    numpy_rng = np.random.RandomState(seed)
    pose_history = []
    last_pose = None
    pose_batch_size = min(batch_size, len(pose_refs))
    pose_bar = tqdm(
        range(pose_steps), desc="pose-pretrain", disable=not verbose)
    for iteration in pose_bar:
        selected = numpy_rng.choice(
            len(pose_refs), pose_batch_size, replace=False)
        images = np.stack([
            frames[pose_refs[item][0]][pose_refs[item][1]]
            for item in selected])
        sensor_params, pose_opt, loss, aux = pose_step(
            sensor_params, pose_opt, jnp.asarray(images))
        last_pose = (loss, *aux)
        if iteration % max(1, pose_steps // TRAINING_HISTORY_POINTS) == 0:
            point = {
                "step": int(iteration + 1),
                "loss": float(loss),
                "reconstruction": float(aux[0]),
                "visual_distance": float(aux[3]),
            }
            pose_history.append(point)
            if callback is not None:
                callback("pose", iteration + 1, sensor_params, point)
        if verbose:
            pose_bar.set_postfix(
                loss=f"{float(loss):.4f}",
                recon=f"{float(aux[0]):.4f}")

    predictor_params = {
        "sensor": sensor_params,
        "A": BOOTSTRAP_A_DECAY * jnp.eye(latent_dim),
        "B": jnp.zeros((latent_dim, control_dim)),
        "c": jnp.zeros(latent_dim),
        "inverse": jnp.zeros(
            (control_dim, 2 * latent_dim)),
        "inverse_c": jnp.zeros(control_dim),
    }
    controlled_tx = optax.adam(optax.cosine_decay_schedule(
        lr, max(int(controlled_steps), 1), alpha=lr_decay_alpha))
    controlled_opt = controlled_tx.init(predictor_params)

    def controlled_objective(params, frame_batch, control_batch, scale):
        batch = frame_batch.shape[0]
        windows = jnp.stack([
            frame_batch[:, step:step + n_frames]
            for step in range(rollout + 1)
        ], axis=1)
        flat_windows = windows.reshape(
            (-1, n_frames, img_size, img_size))
        mean, log_std = model.apply(
            params["sensor"], flat_windows, method=model.encode)
        log_std = jnp.clip(log_std, *LOG_STD_CLIP)
        mean = mean.reshape((batch, rollout + 1, latent_dim))
        log_std = log_std.reshape(
            (batch, rollout + 1, latent_dim))

        pose_total, pose_aux = pose_objective(
            params["sensor"], windows[:, 0, -1])
        prediction = mean[:, 0]
        predictions = []
        for step in range(rollout):
            prediction = (
                prediction @ params["A"].T
                + control_batch[:, step] @ params["B"].T
                + params["c"])
            predictions.append(prediction)
        predictions = jnp.stack(predictions, axis=1)
        target = mean[:, 1:]
        target_log_std = log_std[:, 1:]
        target_variance = jnp.exp(2.0 * target_log_std)
        # The floor must stay far below the anchored latent scale: a floor
        # near the working scale turns this normalization into a constant
        # divisor, and the prediction loss is then minimized by shrinking
        # the encoder output instead of predicting.
        latent_scale = jax.lax.stop_gradient(
            jnp.std(target, axis=(0, 1), keepdims=True) + EPS_LATENT_SCALE)
        prediction_mse = jnp.mean(
            ((predictions - target) / latent_scale) ** 2)
        nll = (
            (predictions - target) ** 2 / target_variance
            + 2.0 * target_log_std)
        # Stop-gradient beta weighting stops the variance head from explaining
        # away hard transitions instead of predicting them.
        beta_weight = jax.lax.stop_gradient(
            target_variance ** beta_nll)
        prediction_nll = 0.5 * jnp.mean(beta_weight * nll)

        inverse_features = jnp.concatenate(
            [mean[:, :-1], mean[:, 1:]], axis=-1)
        predicted_control = (
            jnp.einsum(
                "bti,ui->btu",
                inverse_features, params["inverse"])
            + params["inverse_c"])
        inverse_loss = jnp.mean(
            (predicted_control - control_batch) ** 2)

        rest_frame = windows[:, 0, -1]
        rest_windows = jnp.repeat(
            rest_frame[:, None], n_frames, axis=1)
        rest_mean, _ = model.apply(
            params["sensor"], rest_windows, method=model.encode)
        zero_motion_loss = jnp.mean(
            rest_mean[:, pose_dim:] ** 2)
        rest_target = jnp.concatenate([
            rest_mean[:, :pose_dim],
            jnp.zeros(
                (batch, motion_dim), dtype=rest_mean.dtype),
        ], axis=1)
        rest_next = rest_target @ params["A"].T + params["c"]
        fixed_point_loss = jnp.mean(
            ((rest_next - rest_target)
             / latent_scale.reshape((1, latent_dim))) ** 2)

        state_variance_loss, state_covariance_loss = decorrelation(
            mean.reshape((-1, latent_dim)), normalize=True)
        sensor_variance = jnp.exp(2.0 * log_std)
        sensor_kl = 0.5 * jnp.mean(
            mean ** 2 + sensor_variance
            - 1.0 - 2.0 * log_std)
        dynamics_loss = (
            prediction_weight * prediction_mse
            + prediction_nll_weight * prediction_nll
            + inverse_weight * inverse_loss
            + fixed_point_weight * fixed_point_loss
            + zero_motion_weight * zero_motion_loss
            + PHASE_TWO_DECORRELATION_SCALE * variance_weight * state_variance_loss
            + PHASE_TWO_DECORRELATION_SCALE * covariance_weight * state_covariance_loss
            + kl_weight * sensor_kl)
        total = pose_total + scale * dynamics_loss
        return total, (
            pose_aux[0], pose_aux[1], pose_aux[2], pose_aux[3],
            prediction_mse, prediction_nll, inverse_loss,
            fixed_point_loss, zero_motion_loss,
            state_variance_loss, state_covariance_loss, sensor_kl)

    @jax.jit
    def controlled_step(params, opt_state, frames_batch,
                        actions_batch, scale):
        (loss, aux), gradients = jax.value_and_grad(
            controlled_objective, has_aux=True)(
                params, frames_batch, actions_batch, scale)
        updates, opt_state = controlled_tx.update(
            gradients, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state, loss, aux)

    controlled_history = []
    last_controlled = None
    controlled_batch_size = min(batch_size, len(controlled_refs))
    controlled_bar = tqdm(
        range(controlled_steps), desc="pose-motion", disable=not verbose)
    for iteration in controlled_bar:
        selected = numpy_rng.choice(
            len(controlled_refs), controlled_batch_size, replace=False)
        refs = [controlled_refs[item] for item in selected]
        frame_batch = np.stack([
            frames[sequence][start:start + span]
            for sequence, start in refs])
        control_batch = np.stack([
            controls[sequence][
                start + n_frames - 1:
                start + n_frames - 1 + rollout]
            for sequence, start in refs])
        scale = np.clip(
            (iteration + 1) / max(controlled_ramp, 1),
            0.0, 1.0)
        predictor_params, controlled_opt, loss, aux = controlled_step(
            predictor_params, controlled_opt,
            jnp.asarray(frame_batch), jnp.asarray(control_batch),
            jnp.asarray(scale, dtype=jnp.float32))
        last_controlled = (loss, *aux)
        if iteration % max(1, controlled_steps // TRAINING_HISTORY_POINTS) == 0:
            point = {
                "step": int(iteration + 1),
                "loss": float(loss),
                "reconstruction": float(aux[0]),
                "prediction": float(aux[4]),
                "zero_motion": float(aux[8]),
            }
            controlled_history.append(point)
            if callback is not None:
                callback(
                    "controlled", iteration + 1,
                    predictor_params["sensor"], point)
        if verbose:
            controlled_bar.set_postfix(
                loss=f"{float(loss):.4f}",
                pred=f"{float(aux[4]):.4f}",
                rest=f"{float(aux[8]):.4f}")

    sensor_params = predictor_params["sensor"]
    diagnostics = {
        "architecture": model.architecture(),
        "pose_steps": int(pose_steps),
        "controlled_steps": int(controlled_steps),
        "rollout": int(rollout),
        "history": {
            "pose": pose_history,
            "controlled": controlled_history,
        },
        "bootstrap_A": predictor_params["A"],
        "bootstrap_B": predictor_params["B"],
        "bootstrap_c": predictor_params["c"],
        "control_singular_values": jnp.linalg.svd(
            predictor_params["B"], compute_uv=False),
        "bootstrap_disposable": True,
    }
    if last_pose is not None:
        names = (
            "pose_loss", "pose_reconstruction_loss",
            "pose_variance_loss", "pose_covariance_loss",
            "pose_visual_distance_loss", "pose_kl")
        diagnostics.update({
            name: float(value)
            for name, value in zip(names, last_pose)})
    if last_controlled is not None:
        names = (
            "loss", "reconstruction_loss", "pose_variance_loss_final",
            "pose_covariance_loss_final", "visual_distance_loss",
            "prediction_mse", "prediction_nll", "inverse_loss",
            "fixed_point_loss", "zero_motion_loss",
            "state_variance_loss", "state_covariance_loss", "sensor_kl")
        diagnostics.update({
            name: float(value)
            for name, value in zip(names, last_controlled)})
    return model, sensor_params, diagnostics
