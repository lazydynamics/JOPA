"""Convolutional VAE for 28×28 grayscale frames — a pixel sensor.

The encoder emits ``(μ, log σ)``, which `jopa.VAEObservation` and
`jopa.LearnedVAE` turn into heteroscedastic Gaussian messages. Train it with
`train_vae`, store it with `save_params` / `load_params`.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from tqdm import tqdm

from ..config import LOG_STD_CLIP, PROB_CLIP

SIZE = 28
TRUNK_WIDTH = 256
KERNEL, STRIDE = (4, 4), 2
# exp(-1) ≈ 0.37 starts the sensor moderately uncertain, so the variance head
# neither saturates the clip nor claims precision before the mean head learns.
LOG_STD_BIAS_INIT = -1.0


class _Encoder(nn.Module):
    """``(B, 28, 28)`` → ``(μ, log σ)``, each ``(B, latent_dim)``."""
    latent_dim: int
    ch: int

    @nn.compact
    def __call__(self, x):
        x = x[..., None]
        x = nn.relu(nn.Conv(self.ch, KERNEL, strides=STRIDE, padding="SAME")(x))       # 14
        x = nn.relu(nn.Conv(self.ch * 2, KERNEL, strides=STRIDE, padding="SAME")(x))   # 7
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        mu = nn.Dense(self.latent_dim)(x)
        log_std = nn.Dense(self.latent_dim,
                           bias_init=nn.initializers.constant(LOG_STD_BIAS_INIT))(x)
        return mu, log_std


class _Decoder(nn.Module):
    """``(B, latent_dim)`` → ``(B, 28·28)`` Bernoulli probabilities."""
    ch: int

    @nn.compact
    def __call__(self, z):
        up = (STRIDE, STRIDE)
        x = nn.Dense(TRUNK_WIDTH)(z)
        x = nn.relu(nn.Dense(TRUNK_WIDTH)(x))
        x = nn.relu(nn.Dense(7 * 7 * self.ch * 2)(x))
        x = x.reshape((-1, 7, 7, self.ch * 2))
        x = nn.relu(nn.ConvTranspose(self.ch, KERNEL, strides=up, padding="SAME")(x))  # 14
        x = nn.ConvTranspose(1, KERNEL, strides=up, padding="SAME")(x)                 # 28
        return jax.nn.sigmoid(5.0 * x).reshape((z.shape[0], -1))


class VAE(nn.Module):
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

    def init_params(self, seed=0):
        rng = jax.random.PRNGKey(seed)
        return self.init({"params": rng}, jnp.ones((1, SIZE, SIZE)), rng)


def bernoulli_nll(recon, target):
    """Reconstruction cross-entropy per frame, ``(B,)``."""
    flat = target.reshape(target.shape[0], -1)
    p = jnp.clip(recon, *PROB_CLIP)
    return -jnp.sum(flat * jnp.log(p) + (1 - flat) * jnp.log(1 - p), axis=-1)


def _elbo_loss(params, model, batch, z_rng, beta):
    recon, mu, log_std = model.apply(params, batch, z_rng)
    log_std = jnp.clip(log_std, *LOG_STD_CLIP)
    kl = -0.5 * jnp.sum(1 + 2 * log_std - mu ** 2 - jnp.exp(2 * log_std), axis=-1)
    return jnp.mean(bernoulli_nll(recon, batch) + beta * kl)


def train_vae(images, *, latent_dim=2, ch=32, epochs=200, batch_size=64, lr=1e-3,
              beta=1.0, beta_start=0.1, beta_warmup=15, seed=0, verbose=True,
              callback: Callable | None = None) -> tuple[VAE, dict]:
    """Train on ``images`` ``(N, 28, 28)`` by the ELBO. The KL weight anneals
    ``beta_start`` → ``beta`` over the first ``beta_warmup`` epochs, so the
    encoder locks on before the posterior can collapse."""
    images = jnp.asarray(images)
    if images.shape[1:] != (SIZE, SIZE):
        raise ValueError(f"expected (N, {SIZE}, {SIZE}) frames, got {images.shape}")
    model = VAE(latent_dim=latent_dim, ch=ch)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, z_rng = jax.random.split(rng, 3)
    params = model.init({"params": init_rng}, jnp.ones((1, SIZE, SIZE)), z_rng)
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, batch, z_rng, beta):
        loss, grads = jax.value_and_grad(_elbo_loss)(params, model, batch, z_rng, beta)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    n = images.shape[0]
    pbar = tqdm(range(1, epochs + 1), desc="VAE", disable=not verbose)
    for epoch in pbar:
        rng, perm_rng = jax.random.split(rng)
        imgs = images[jax.random.permutation(perm_rng, n)]
        beta_t = min(beta, beta_start + (beta - beta_start) * (epoch - 1) / max(beta_warmup, 1))
        losses = []
        for i in range(0, n, batch_size):
            rng, z_rng = jax.random.split(rng)
            params, opt_state, loss = step(params, opt_state, imgs[i:i + batch_size],
                                           z_rng, beta_t)
            losses.append(float(loss))
        pbar.set_postfix(loss=f"{np.mean(losses):.1f}", beta=f"{beta_t:.2f}")
        if callback is not None:
            callback(epoch, params, float(np.mean(losses)))
    return model, params


def save_params(params, path: str | Path):
    with open(str(path), "wb") as f:
        f.write(serialization.to_bytes(params))


def load_params(model: VAE, path: str | Path) -> dict:
    """Read a `save_params` checkpoint; rejects one written for another shape."""
    template = model.init_params()
    with open(str(path), "rb") as f:
        loaded = serialization.from_bytes(template, f.read())
    def shapes(tree):
        return [np.shape(x) for x in jax.tree.leaves(tree)]
    if (jax.tree.structure(loaded) != jax.tree.structure(template)
            or shapes(loaded) != shapes(template)):
        raise ValueError(f"checkpoint {path} does not match {model}")
    return loaded
