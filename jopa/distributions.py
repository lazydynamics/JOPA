"""Probability distributions for message passing in natural parameter form."""
from typing import NamedTuple
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Distribution types (JAX pytree-compatible via NamedTuple)
# ---------------------------------------------------------------------------

class Gaussian(NamedTuple):
    """Multivariate normal in information (natural) form.

    η = Λμ   (weighted mean / precision-weighted mean)
    Λ = Σ⁻¹  (precision matrix)
    """
    eta: jnp.ndarray  # (d,)
    lam: jnp.ndarray  # (d, d)


class Wishart(NamedTuple):
    """Wishart distribution in natural parameter form.

    density ∝ |W|^{(df-d-1)/2} exp(-½ tr(inv_scale · W))
    mean = df · inv(inv_scale)
    """
    df: float              # degrees of freedom  ν
    inv_scale: jnp.ndarray  # V⁻¹  (d, d)


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------

def gaussian_mean(g: Gaussian) -> jnp.ndarray:
    return jnp.linalg.solve(g.lam, g.eta)


def gaussian_mean_cov(g: Gaussian):
    cov = jnp.linalg.inv(g.lam)
    return cov @ g.eta, cov


def combine_gaussians(*gs: Gaussian) -> Gaussian:
    """Product of Gaussian messages = sum of natural parameters."""
    eta = gs[0].eta
    lam = gs[0].lam
    for g in gs[1:]:
        eta = eta + g.eta
        lam = lam + g.lam
    return Gaussian(eta=eta, lam=lam)


def gaussian_prior(dim: int, cov_scale: float, mean: jnp.ndarray | None = None) -> Gaussian:
    """Create an isotropic Gaussian prior in information form."""
    lam = (1.0 / cov_scale) * jnp.eye(dim)
    eta = lam @ mean if mean is not None else jnp.zeros(dim)
    return Gaussian(eta=eta, lam=lam)


def near_identity_prior(latent_dim: int, cov: float = 0.5) -> dict:
    """Bayesian-prior kwargs centring ``vec(A)`` on the identity matrix.

    A soft "roughly identity dynamics" assumption — useful when you expect
    small per-step changes in the latent state. The same ``cov`` is reused
    for the initial posterior, since the data drives the result quickly.

    Returns
    -------
    dict
        ``{prior_a_mean, prior_a_cov, init_a_mean, init_a_cov}`` — drop into
        :func:`jopa.inference.infer` or :func:`jopa.em.variational_em` with
        ``**``-unpacking::

            result = variational_em(..., **near_identity_prior(d, cov=0.5))
    """
    vec_I = jnp.eye(latent_dim).ravel()
    return {
        "prior_a_mean": vec_I,
        "prior_a_cov": cov,
        "init_a_mean": vec_I,
        "init_a_cov": cov,
    }


def vague_gaussian(d: int) -> Gaussian:
    """Uninformative (zero-precision) Gaussian message."""
    return Gaussian(eta=jnp.zeros(d), lam=jnp.zeros((d, d)))


# ---------------------------------------------------------------------------
# Wishart helpers
# ---------------------------------------------------------------------------

def wishart_mean(w: Wishart) -> jnp.ndarray:
    """E[W] = df · scale = df · inv(inv_scale)."""
    return w.df * jnp.linalg.inv(w.inv_scale)
