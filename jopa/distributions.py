"""Gaussian / Wishart distributions in natural parameter form."""
from typing import NamedTuple
import jax.numpy as jnp


class Gaussian(NamedTuple):
    """Multivariate normal in information form:  η = Λμ,  Λ = Σ⁻¹."""
    eta: jnp.ndarray  # (d,) or (T, d)
    lam: jnp.ndarray  # (d, d) or (T, d, d)


class Wishart(NamedTuple):
    """Wishart in natural form:  mean = df · inv(inv_scale)."""
    df: float
    inv_scale: jnp.ndarray  # (d, d)


# ---- Gaussian helpers ------------------------------------------------------

def gaussian_mean(g: Gaussian) -> jnp.ndarray:
    return jnp.linalg.solve(g.lam, g.eta)


def gaussian_mean_cov(g: Gaussian):
    cov = jnp.linalg.inv(g.lam)
    return cov @ g.eta, cov


def combine_gaussians(*gs: Gaussian) -> Gaussian:
    """Product of Gaussian messages — sum of natural parameters."""
    eta = sum(g.eta for g in gs)
    lam = sum(g.lam for g in gs)
    return Gaussian(eta=eta, lam=lam)


def gaussian_prior(dim: int, cov_scale: float, mean: jnp.ndarray | None = None) -> Gaussian:
    """Isotropic Gaussian prior in information form."""
    lam = (1.0 / cov_scale) * jnp.eye(dim)
    eta = lam @ mean if mean is not None else jnp.zeros(dim)
    return Gaussian(eta=eta, lam=lam)


def vague_gaussian(d: int) -> Gaussian:
    """Uninformative (zero-precision) Gaussian message."""
    return Gaussian(eta=jnp.zeros(d), lam=jnp.zeros((d, d)))


def near_identity_prior(latent_dim: int, cov: float = 0.5) -> dict:
    """Kwargs centring vec(A) on the identity matrix — useful when you expect
    small per-step changes in the latent state. Drop into LearnedLinear with
    ``**``-unpacking."""
    vec_I = jnp.eye(latent_dim).ravel()
    return {
        "prior_a_mean": vec_I,
        "prior_a_cov": cov,
        "init_a_cov": cov,
    }


# ---- Wishart helpers -------------------------------------------------------

def wishart_mean(w: Wishart) -> jnp.ndarray:
    return w.df * jnp.linalg.inv(w.inv_scale)
