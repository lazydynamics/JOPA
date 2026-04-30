"""Convolutional VAE for learning smooth latent representations of spatial data."""
from __future__ import annotations
from typing import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm


# Numerical guards used by both the VAE ELBO and the Variational-EM M-step.
LOG_STD_CLIP = (-6.0, 2.0)         # encoder output stability
PROB_CLIP = (1e-6, 1.0 - 1e-6)      # decoder Bernoulli probabilities


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Strided-conv encoder  (B, 28, 28) → (μ, log σ) each (B, d)."""
    latent_dim: int = 2
    ch: int = 32  # base channel width

    @nn.compact
    def __call__(self, x):
        c = self.ch
        x = x.reshape((-1, 28, 28, 1))
        x = nn.relu(nn.Conv(c,     (4, 4), strides=2, padding="SAME")(x))   # 14×14
        x = nn.relu(nn.Conv(c * 2, (4, 4), strides=2, padding="SAME")(x))   #  7×7
        x = x.reshape((x.shape[0], -1))                                      # 3136
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(256)(x))
        mu      = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(self.latent_dim,
                           bias_init=nn.initializers.constant(-1.0))(x)
        return mu, log_std


class _Decoder(nn.Module):
    """Transposed-conv decoder  (B, d) → (B, 784) Bernoulli probs."""
    ch: int = 32

    @nn.compact
    def __call__(self, z):
        c = self.ch
        x = nn.Dense(256)(z)
        x = nn.relu(nn.Dense(256)(x))
        x = nn.relu(nn.Dense(7 * 7 * c * 2)(x))
        x = x.reshape((-1, 7, 7, c * 2))
        x = nn.relu(nn.ConvTranspose(c, (4, 4), strides=(2, 2), padding="SAME")(x))  # 14×14
        x = nn.ConvTranspose(1, (4, 4), strides=(2, 2), padding="SAME")(x)            # 28×28
        # Scaled sigmoid — sharper outputs for binarised data
        return jax.nn.sigmoid(5.0 * x).reshape((z.shape[0], -1))


class VAE(nn.Module):
    """Convolutional VAE for 28×28 images with a low-dimensional latent space."""
    latent_dim: int = 2
    ch: int = 32

    def setup(self):
        self.encoder = _Encoder(self.latent_dim, self.ch)
        self.decoder = _Decoder(self.ch)

    def __call__(self, x, z_rng):
        mu, log_std = self.encoder(x)
        z = mu + jnp.exp(log_std) * jax.random.normal(z_rng, mu.shape)
        return self.decoder(z), mu, log_std

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _elbo_loss(params, model, batch, z_rng, beta):
    recon, mu, log_std = model.apply(params, batch, z_rng)
    log_std = jnp.clip(log_std, *LOG_STD_CLIP)
    flat = batch.reshape(batch.shape[0], -1)
    p = jnp.clip(recon, *PROB_CLIP)
    bce = -jnp.sum(flat * jnp.log(p) + (1 - flat) * jnp.log(1 - p), axis=-1)
    kl = -0.5 * jnp.sum(1 + 2 * log_std - mu ** 2 - jnp.exp(2 * log_std), axis=-1)
    return jnp.mean(bce + beta * kl)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_vae(
    images: np.ndarray,
    *,
    latent_dim: int = 2,
    ch: int = 32,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
    callback: Callable | None = None,
) -> tuple[VAE, dict]:
    """Train a ConvVAE on (N, 28, 28) images in [0, 1].

    β-annealing from 0.1 → 1.0 over first 15 epochs for stable training.
    """
    model = VAE(latent_dim=latent_dim, ch=ch)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, z_rng = jax.random.split(rng, 3)
    params = model.init({"params": init_rng}, jnp.ones((1, 28, 28)), z_rng)

    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch, z_rng, beta):
        loss, grads = jax.value_and_grad(_elbo_loss)(
            params, model, batch, z_rng, beta,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = images.shape[0]
    pbar = tqdm(range(1, epochs + 1), desc="VAE", disable=not verbose)
    for epoch in pbar:
        rng, perm_rng = jax.random.split(rng)
        imgs = images[jax.random.permutation(perm_rng, n)]
        beta = min(1.0, 0.1 + 0.9 * (epoch - 1) / 15)

        losses = []
        for i in range(0, n, batch_size):
            rng, z_rng = jax.random.split(rng)
            params, opt_state, loss = step(
                params, opt_state, jnp.array(imgs[i : i + batch_size]), z_rng, beta,
            )
            losses.append(float(loss))

        pbar.set_postfix(loss=f"{np.mean(losses):.1f}", beta=f"{beta:.2f}")

        if callback is not None:
            callback(epoch, params, float(np.mean(losses)))

    return model, params


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------

def make_encode_decode(model, params):
    """Create jitted single-image encode/decode functions from a VAE."""

    @jax.jit
    def encode_fn(image):
        mu, ls = model.apply(params, image.reshape(1, 28, 28), method=model.encode)
        return mu[0], ls[0]

    @jax.jit
    def decode_fn(z):
        return model.apply(params, z.reshape(1, -1), method=model.decode)[0].reshape(28, 28)

    return encode_fn, decode_fn


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_params(params, path: str | Path):
    flat = jax.tree.leaves(params)
    np.savez(str(path), **{f"p{i}": np.asarray(v) for i, v in enumerate(flat)})


def load_params(model: VAE, path: str | Path) -> dict:
    data = np.load(str(path))
    rng = jax.random.PRNGKey(0)
    params = model.init({"params": rng}, jnp.ones((1, 28, 28)), rng)
    flat = [jnp.array(data[f"p{i}"]) for i in range(len(jax.tree.leaves(params)))]
    return jax.tree.unflatten(jax.tree.structure(params), flat)
