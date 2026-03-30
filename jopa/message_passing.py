"""Belief propagation and VMP utilities for latent chain models.

Shared forward-backward pass and message accumulation used by both
single-sequence inference and multi-trajectory Variational EM.
"""
import jax
import jax.numpy as jnp

from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, vague_gaussian,
)
from .nodes.transition import (
    CTCache,
    ct_forward, ct_backward, ct_marginal_yx,
    ct_message_a, ct_message_b, ct_message_W,
)


def encode_observations(observations, encode_fn, latent_dim):
    """Convert a sequence of observations to VAE messages.

    None entries become vague (uninformative) Gaussian messages.
    Uses batched encoding for efficiency.
    """
    T = len(observations)
    present_idx = [i for i in range(T) if observations[i] is not None]

    if not present_idx:
        return [vague_gaussian(latent_dim)] * T

    images = jnp.stack([jnp.array(observations[i]) for i in present_idx])
    means, log_stds = jax.vmap(encode_fn)(images)

    msgs = [vague_gaussian(latent_dim)] * T
    for j, i in enumerate(present_idx):
        var = jnp.exp(2.0 * log_stds[j])
        lam = jnp.diag(1.0 / var)
        eta = lam @ means[j]
        msgs[i] = Gaussian(eta=eta, lam=lam)
    return msgs


def forward_backward(prior_x, vae_msgs, cache, actions=None):
    """Run forward-backward belief propagation on a latent chain.

    Parameters
    ----------
    prior_x : Gaussian — prior on x[0]
    vae_msgs : list of Gaussian, length T — observation messages
    cache : CTCache — transition node cache
    actions : list, length T-1, optional
        Control inputs per transition. None = no control.

    Returns
    -------
    alphas : list of Gaussian, length T — forward messages
    betas : list of Gaussian, length T — backward messages
    m_xs : list of Gaussian, length T-1 — input-side combined messages
    m_ys : list of Gaussian, length T-1 — output-side combined messages
    """
    T = len(vae_msgs)
    d = prior_x.eta.shape[0]

    # Forward
    alphas = [prior_x]
    m_xs = []
    for t in range(1, T):
        m_x = combine_gaussians(alphas[t - 1], vae_msgs[t - 1])
        m_xs.append(m_x)
        u = actions[t - 1] if actions is not None else None
        alphas.append(ct_forward(m_x, cache, u))

    # Backward
    betas = [None] * T
    betas[T - 1] = vague_gaussian(d)
    m_ys = [None] * (T - 1)
    for t in range(T - 1, 0, -1):
        m_y = combine_gaussians(betas[t], vae_msgs[t])
        m_ys[t - 1] = m_y
        u = actions[t - 1] if actions is not None else None
        betas[t - 1] = ct_backward(m_y, cache, u)

    return alphas, betas, m_xs, m_ys


def accumulate_vmp_messages(m_xs, m_ys, cache, actions,
                            prior_a, prior_W, prior_b=None):
    """Accumulate VMP messages for transition parameters from BP marginals.

    Returns updated posteriors (q_a, q_W, q_b).
    """
    n_ct = len(m_xs)
    acc_a_eta = prior_a.eta.copy()
    acc_a_lam = prior_a.lam.copy()
    acc_W_inv = prior_W.inv_scale.copy()

    has_control = prior_b is not None
    if has_control:
        acc_b_eta = prior_b.eta.copy()
        acc_b_lam = prior_b.lam.copy()

    for k in range(n_ct):
        u_k = actions[k] if actions is not None else None
        q_yx = ct_marginal_yx(m_ys[k], m_xs[k], cache, u_k)
        msg_a = ct_message_a(q_yx, cache, u_k)
        delta = ct_message_W(q_yx, cache, u_k)
        acc_a_eta = acc_a_eta + msg_a.eta
        acc_a_lam = acc_a_lam + msg_a.lam
        acc_W_inv = acc_W_inv + delta
        if has_control:
            msg_b = ct_message_b(q_yx, cache, u_k)
            acc_b_eta = acc_b_eta + msg_b.eta
            acc_b_lam = acc_b_lam + msg_b.lam

    q_a = Gaussian(eta=acc_a_eta, lam=acc_a_lam)
    q_W = Wishart(df=prior_W.df + n_ct, inv_scale=acc_W_inv)
    q_b = Gaussian(eta=acc_b_eta, lam=acc_b_lam) if has_control else None
    return q_a, q_W, q_b


def compute_marginals(alphas, betas, vae_msgs):
    """Compute marginal posteriors q(x[t]) from BP messages."""
    return [combine_gaussians(alphas[t], betas[t], vae_msgs[t])
            for t in range(len(alphas))]
