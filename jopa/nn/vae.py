"""Convolutional VAE for square grayscale frames (28, 64 or 128 px).

`n_frames=1` is the plain single-frame VAE. `n_frames=K>1` stacks K consecutive
frames as encoder input channels and reconstructs the whole K-frame window, so
the latent must encode motion (velocity), not just the latest pose.

This module also re-exports the rest of `jopa.nn` so that
`from jopa.nn.vae import ...` keeps working for every symbol it ever exposed.
"""
from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from ..config import (
    DEFAULT_IMG_SIZE,
    DEFAULT_LATENT_DIM,
    DEFAULT_N_FRAMES,
    ENCODER_CHANNELS,
    LOG_STD_CLIP,
    PROB_CLIP,
    SUPPORTED_IMG_SIZES,
    VAETrainingConfig,
)
from .layers import _as_batch, _Decoder, _Encoder, _latest_frame
from .losses import decorrelation
from .persistence import (
    VAEAdapter,
    _validate_checkpoint_tree,
    load_params,
    make_encode_decode,
    save_params,
)
from .pose_motion import PoseMotionVAE, _MotionHead, train_pose_motion_vae

__all__ = [
    "LOG_STD_CLIP",
    "PROB_CLIP",
    "VAE",
    "PoseMotionVAE",
    "VAEAdapter",
    "_Decoder",
    "_Encoder",
    "_MotionHead",
    "_as_batch",
    "_latest_frame",
    "_validate_checkpoint_tree",
    "decorrelation",
    "load_params",
    "make_encode_decode",
    "save_params",
    "train_pose_motion_vae",
    "train_vae",
]


class VAE(nn.Module):
    """Convolutional VAE for square grayscale frames (`img_size` 28, 64 or 128).

    `n_frames=1` (default): single-frame encoder + single-frame decoder.
    `n_frames=K>1`: encoder sees K stacked frames and the decoder reconstructs
    the whole K-frame window, so the latent must encode motion (velocity), not
    just the latest pose.
    """
    latent_dim: int = DEFAULT_LATENT_DIM
    ch: int = ENCODER_CHANNELS
    n_frames: int = DEFAULT_N_FRAMES
    img_size: int = DEFAULT_IMG_SIZE

    def setup(self):
        if self.img_size not in SUPPORTED_IMG_SIZES:
            raise ValueError(f"img_size must be 28, 64 or 128, got {self.img_size}")
        self.encoder = _Encoder(self.latent_dim, self.ch, self.n_frames, self.img_size)
        self.decoder = _Decoder(self.ch, self.n_frames, self.img_size)

    def __call__(self, x, z_rng):
        mu, log_std = self.encoder(x)
        z = mu + jnp.exp(log_std) * jax.random.normal(z_rng, mu.shape)
        return self.decoder(z), mu, log_std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def _elbo_loss(params, model, batch, target, z_rng, beta):
    """ELBO. `batch` is the encoder input (window or single frame); `target`
    is what the decoder must reconstruct — the full input (the whole K-frame
    window for multi-frame VAEs)."""
    recon, mu, log_std = model.apply(params, batch, z_rng)
    log_std = jnp.clip(log_std, *LOG_STD_CLIP)
    flat = target.reshape(target.shape[0], -1)
    p = jnp.clip(recon, *PROB_CLIP)
    bce = -jnp.sum(flat * jnp.log(p) + (1 - flat) * jnp.log(1 - p), axis=-1)
    kl = -0.5 * jnp.sum(1 + 2 * log_std - mu ** 2 - jnp.exp(2 * log_std), axis=-1)
    return jnp.mean(bce + beta * kl)


def train_vae(
    images: np.ndarray,
    *,
    config: VAETrainingConfig | None = None,
    latent_dim: int = VAETrainingConfig.latent_dim,
    ch: int = VAETrainingConfig.ch,
    n_frames: int = VAETrainingConfig.n_frames,
    img_size: int = VAETrainingConfig.img_size,
    epochs: int = VAETrainingConfig.epochs,
    batch_size: int = VAETrainingConfig.batch_size,
    lr: float = VAETrainingConfig.lr,
    beta: float = VAETrainingConfig.beta,
    beta_start: float = VAETrainingConfig.beta_start,
    beta_warmup: int = VAETrainingConfig.beta_warmup,
    seed: int = VAETrainingConfig.seed,
    verbose: bool = True,
    callback: Callable | None = None,
) -> tuple[VAE, dict]:
    """Train the VAE (autoencoder ELBO — the decoder reconstructs its input).

    Single-frame: `images` of shape `(N, S, S)` with `S = img_size`.
    Multi-frame: `images` of shape `(N, K, S, S)` (the windows); the decoder
    reconstructs the whole K-frame window, so the latent must encode motion.
    `n_frames` must equal K.

    β anneals `beta_start` → `beta` over the first `beta_warmup` epochs.
    Large low-contrast frames need a long, low warmup (e.g. `beta_start=0.0,
    beta_warmup=80`) or the posterior collapses before the encoder locks on.

    Hyperparameters are documented as the fields of
    :class:`jopa.config.VAETrainingConfig`; passing a preset as ``config``
    supersedes the individual keywords.
    """
    if config is not None:
        return train_vae(
            images, verbose=verbose, callback=callback, **config.kwargs())
    images = jnp.asarray(images)
    S = img_size
    if n_frames == 1:
        if images.ndim == 4:
            if images.shape[1] != 1:
                raise ValueError(
                    f"single-frame training expects (N, {S}, {S}) or (N, 1, {S}, {S}), got {images.shape}")
            images = images[:, 0]
        elif images.ndim != 3:
            raise ValueError(
                f"single-frame training expects (N, {S}, {S}), got {images.shape}")
    elif images.ndim != 4 or images.shape[1] != n_frames:
        raise ValueError(
            f"n_frames={n_frames} expects (N, {n_frames}, {S}, {S}), got {images.shape}")
    if images.shape[-1] != S or images.shape[-2] != S:
        raise ValueError(f"img_size={S} expects {S}×{S} frames, got {images.shape}")

    model = VAE(latent_dim=latent_dim, ch=ch, n_frames=n_frames, img_size=S)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, z_rng = jax.random.split(rng, 3)
    init_input = (jnp.ones((1, S, S)) if n_frames == 1
                  else jnp.ones((1, n_frames, S, S)))
    params = model.init({"params": init_rng}, init_input, z_rng)

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch, target, z_rng, beta):
        loss, grads = jax.value_and_grad(_elbo_loss)(
            params, model, batch, target, z_rng, beta,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = images.shape[0]
    pbar = tqdm(range(1, epochs + 1), desc="VAE", disable=not verbose)
    for epoch in pbar:
        rng, perm_rng = jax.random.split(rng)
        idx = jax.random.permutation(perm_rng, n)
        imgs = images[idx]
        beta_t = min(beta, beta_start + (beta - beta_start) * (epoch - 1) / beta_warmup)
        losses = []
        for i in range(0, n, batch_size):
            rng, z_rng = jax.random.split(rng)
            batch = imgs[i : i + batch_size]
            params, opt_state, loss = step(
                params, opt_state, batch, batch, z_rng, beta_t,
            )
            losses.append(float(loss))
        pbar.set_postfix(loss=f"{np.mean(losses):.1f}", beta=f"{beta_t:.2f}")
        if callback is not None:
            callback(epoch, params, float(np.mean(losses)))
    return model, params
