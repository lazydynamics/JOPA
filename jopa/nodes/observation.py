"""VAE observation node for bridging neural networks with message passing.

Forward rule (y ← x): decode latent mean → predicted image.

The backward rule (image → Gaussian message) is provided by
``jopa.message_passing.encode_observations`` for whole sequences at once.
"""
import jax.numpy as jnp
from ..distributions import Gaussian


def vae_predict(q_x: Gaussian, decode_fn) -> jnp.ndarray:
    """Forward prediction: decode latent posterior mean into an image.

    Parameters
    ----------
    q_x : Gaussian
        Posterior over latent state.
    decode_fn : callable
        z → image  where z has shape (d,).

    Returns
    -------
    Predicted image array.
    """
    mu = jnp.linalg.solve(q_x.lam, q_x.eta)
    return decode_fn(mu)
