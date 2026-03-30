"""VAE observation node for bridging neural networks with message passing.

Backward rule (x ← y): encode observed image → Gaussian message in latent space.
Forward  rule (y ← x): decode latent mean   → predicted image.
"""
import jax.numpy as jnp
from ..distributions import Gaussian


def vae_observe(image: jnp.ndarray, encode_fn) -> Gaussian:
    """Backward message: encode an observed image into a latent Gaussian.

    Parameters
    ----------
    image : array (H, W) or (H, W, 1)
        Observed image.
    encode_fn : callable
        image → (mean, log_std)  where both have shape (d,).

    Returns
    -------
    Gaussian in information form.
    """
    mean, log_std = encode_fn(image)
    var = jnp.exp(2.0 * log_std)
    lam = jnp.diag(1.0 / var)
    eta = lam @ mean
    return Gaussian(eta=eta, lam=lam)


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
