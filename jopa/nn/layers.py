"""Shared convolutional trunks, latent clips and frame-shape helpers.

The 28×28 layer structure is fixed: changing it invalidates checkpoints saved
before `img_size` existed.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..config import (
    CONV_KERNEL,
    CONV_STRIDE,
    DECODER_BASE_SIZE_LARGE,
    DECODER_BASE_SIZE_SMALL,
    DECODER_LOGIT_GAIN,
    DEFAULT_IMG_SIZE,
    DEFAULT_LATENT_DIM,
    DEFAULT_N_FRAMES,
    ENCODER_CHANNELS,
    IMG_SIZE_LARGE,
    IMG_SIZE_MEDIUM,
    IMG_SIZE_SMALL,
    LOG_STD_BIAS_INIT,
    LOG_STD_CLIP,
    PROB_CLIP,
    TRUNK_WIDTH,
)

__all__ = ["LOG_STD_CLIP", "PROB_CLIP"]


def _as_batch(window, n_frames, img_size=DEFAULT_IMG_SIZE):
    """A single (S,S) frame or (K,S,S) window → a 1-element model batch."""
    w = jnp.asarray(window)
    return (w.reshape(1, img_size, img_size) if n_frames == 1
            else w.reshape(1, n_frames, img_size, img_size))


def _latest_frame(recon, img_size=DEFAULT_IMG_SIZE):
    """The most-recent S×S frame from a flat decoder output (… → S×S)."""
    return jnp.asarray(recon).reshape(-1, img_size, img_size)[-1]


class _Encoder(nn.Module):
    """Strided-conv encoder.  Input `(B, K, S, S)` (K = n_frames stacked as
    input channels) or `(B, S, S)` when K=1.  Output `(μ, log σ)` each `(B, d)`."""
    latent_dim: int = DEFAULT_LATENT_DIM
    ch: int = ENCODER_CHANNELS
    n_frames: int = DEFAULT_N_FRAMES
    img_size: int = DEFAULT_IMG_SIZE

    @nn.compact
    def __call__(self, x):
        c = self.ch
        k, s = CONV_KERNEL, CONV_STRIDE
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = jnp.transpose(x, (0, 2, 3, 1))                # (B, S, S, K)
        if self.img_size >= IMG_SIZE_MEDIUM:
            x = 2.0 * x - 1.0                             # centering keeps the deeper stack from starving
        x = nn.relu(nn.Conv(c,     k, strides=s, padding="SAME")(x))   # S/2
        x = nn.relu(nn.Conv(c * 2, k, strides=s, padding="SAME")(x))   # S/4
        if self.img_size >= IMG_SIZE_MEDIUM:                           # → 8×8
            x = nn.relu(nn.Conv(c * 2, k, strides=s, padding="SAME")(x))
        if self.img_size == IMG_SIZE_LARGE:
            x = nn.relu(nn.Conv(c * 2, k, strides=s, padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        mu      = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(
            self.latent_dim,
            bias_init=nn.initializers.constant(LOG_STD_BIAS_INIT))(x)
        return mu, log_std


class _Decoder(nn.Module):
    """Transposed-conv decoder  `(B, d) → (B, n_frames*S*S)` Bernoulli probs.
    `n_frames=1` (default) emits one S×S frame; `n_frames=K>1` reconstructs
    the whole K-frame window, which forces the latent to encode motion
    (velocity), not just the latest pose."""
    ch: int = ENCODER_CHANNELS
    n_frames: int = DEFAULT_N_FRAMES
    img_size: int = DEFAULT_IMG_SIZE

    @nn.compact
    def __call__(self, z):
        c = self.ch
        k, up = CONV_KERNEL, (CONV_STRIDE, CONV_STRIDE)
        base = (DECODER_BASE_SIZE_SMALL if self.img_size == IMG_SIZE_SMALL
                else DECODER_BASE_SIZE_LARGE)
        x = nn.Dense(TRUNK_WIDTH)(z)
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        x = nn.relu(nn.Dense(base * base * c * 2)(x))
        x = x.reshape((-1, base, base, c * 2))
        if self.img_size >= IMG_SIZE_MEDIUM:                                     # 8 → 16
            x = nn.relu(nn.ConvTranspose(c * 2, k, strides=up, padding="SAME")(x))
        if self.img_size == IMG_SIZE_LARGE:                                      # 16 → 32
            x = nn.relu(nn.ConvTranspose(c * 2, k, strides=up, padding="SAME")(x))
        x = nn.relu(nn.ConvTranspose(c, k, strides=up, padding="SAME")(x))       # S/2
        x = nn.ConvTranspose(self.n_frames, k, strides=up, padding="SAME")(x)    # S×S×K
        x = jnp.transpose(x, (0, 3, 1, 2))                                       # (B, K, S, S)
        return jax.nn.sigmoid(DECODER_LOGIT_GAIN * x).reshape((z.shape[0], -1))
