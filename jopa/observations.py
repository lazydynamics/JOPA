"""Observations: raw datum → Gaussian message on a block's state slice.

Frozen sensors (`Frozen`, `VAEObservation`) only
encode; `LearnedVAE` also refines its weights in the M-step.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from .config import LOG_STD_CLIP
from .distributions import Gaussian
from .nn.vae import SIZE, bernoulli_nll


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
    """ELBO for the VAE M-step: reconstruction + KL(encoder ‖ smoothed posterior)."""
    mu_enc, log_std_enc = model.apply(params, images, method=model.encode)
    log_std_enc = jnp.clip(log_std_enc, *LOG_STD_CLIP)
    z = mu_enc + jnp.exp(log_std_enc) * jax.random.normal(z_rng, mu_enc.shape)
    recon = model.apply(params, z, method=model.decode)
    kl = jax.vmap(_kl_diag_vs_full)(mu_enc, log_std_enc, post_means, post_covs)
    return jnp.mean(beta_recon * bernoulli_nll(recon, images) + kl)


def _message(mu, log_std) -> Gaussian:
    lam = jnp.diag(jnp.exp(-2.0 * jnp.clip(log_std, *LOG_STD_CLIP)))
    return Gaussian(eta=lam @ mu, lam=lam)


class LearnedVAE(Observation):
    """VAE observation (`jopa.nn.VAE`) whose encoder produces messages and
    whose weights are refined in the M-step against the smoothed posterior."""
    learnable = True

    def __init__(self, model, params, lr=5e-5, n_m_steps=20, beta_recon=1.0, seed=0):
        self.model = model
        self.params = params
        self.n_m_steps = n_m_steps
        self.beta_recon = beta_recon
        self._tx = optax.adam(lr)
        self._opt_state = self._tx.init(params)
        self._rng = jax.random.PRNGKey(seed)
        self.loss_history: list[float] = []

    def message(self, image) -> Gaussian:
        mu, log_std = self.model.apply(self.params, jnp.reshape(image, (1, SIZE, SIZE)),
                                       method=self.model.encode)
        return _message(mu[0], log_std[0])

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
        return self.model.apply(self.params, jnp.reshape(z, (1, -1)),
                                method=self.model.decode).reshape(SIZE, SIZE)


class VAEObservation(Observation):
    """Frozen VAE sensor (`jopa.nn.VAE`): a frame becomes a Gaussian message
    whose precision the encoder sets per frame and per dimension."""
    learnable = False

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self._encode = jax.jit(lambda p, x: model.apply(p, x, method=model.encode))
        self._decode = jax.jit(lambda p, z: model.apply(p, z, method=model.decode))

    def message(self, image) -> Gaussian:
        mu, log_std = self._encode(self.params, jnp.reshape(image, (1, SIZE, SIZE)))
        return _message(mu[0], log_std[0])

    def decode(self, z):
        return self._decode(self.params, jnp.reshape(z, (1, -1))).reshape(SIZE, SIZE)


def _as_observation(obj) -> Observation:
    return obj if isinstance(obj, Observation) else Frozen(obj)
