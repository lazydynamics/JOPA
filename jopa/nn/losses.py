"""Anti-collapse regularizers shared by the pixel-only sensor trainers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import EPS_VARIANCE


def decorrelation(features, *, normalize=False):
    """Per-dimension variance anchor + off-diagonal covariance penalty.

    The eps must stay far below any reachable batch variance and the hinge
    needs the log term: `relu(1 - sqrt(var + eps))` alone loses its restoring
    gradient once `var < eps`, so near-constant features (e.g. frames dominated
    by a static background) collapse to zero scale.

    `normalize=True` penalizes correlations rather than covariances, making the
    off-diagonal term independent of the anchored feature scale.
    """
    std = jnp.sqrt(jnp.var(features, axis=0) + EPS_VARIANCE)
    variance_loss = jnp.mean(
        jax.nn.relu(1.0 - std) + jax.nn.relu(-jnp.log(std)))
    centered = features - jnp.mean(features, axis=0)
    if normalize:
        centered = centered / jnp.maximum(std, 1e-3)
    covariance = centered.T @ centered / jnp.maximum(
        features.shape[0] - 1, 1)
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    covariance_loss = jnp.sum(off_diagonal ** 2) / features.shape[1]
    return variance_loss, covariance_loss
