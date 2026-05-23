"""Belief propagation and VMP utilities for latent chain models.

Shared forward-backward pass and message accumulation used by both
single-sequence inference and multi-trajectory Variational EM.

Performance
-----------
The inner loops run under ``lax.scan`` / ``vmap`` with the full sequence
stacked as one pytree, so the bulk of the work for a sequence of length T
fuses into a single JIT'd graph rather than T separate kernel launches.
Public functions accept ``list[Gaussian]`` for ergonomics; sequences are
stacked at the boundary.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, vague_gaussian,
)
from .nodes.transition import (
    ct_forward, ct_backward, ct_marginal_yx,
    ct_message_a, ct_message_b, ct_message_W,
)


# ---------------------------------------------------------------------------
# Stacking helpers — list<Gaussian> ↔ Gaussian(eta:(T,d), lam:(T,d,d))
# ---------------------------------------------------------------------------

def _stack_gaussians(msgs):
    """Stack a list of Gaussians into one Gaussian whose leaves carry a
    leading time axis."""
    return Gaussian(
        eta=jnp.stack([m.eta for m in msgs]),
        lam=jnp.stack([m.lam for m in msgs]),
    )


# ---------------------------------------------------------------------------
# Observation encoding (unchanged interface; already vmaps the VAE encoder)
# ---------------------------------------------------------------------------

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

    msgs = [vague_gaussian(latent_dim) for _ in range(T)]
    for j, i in enumerate(present_idx):
        var = jnp.exp(2.0 * log_stds[j])
        lam = jnp.diag(1.0 / var)
        eta = lam @ means[j]
        msgs[i] = Gaussian(eta=eta, lam=lam)
    return msgs


# ---------------------------------------------------------------------------
# Forward-backward via lax.scan
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("has_control",))
def _fb_scan(prior_x, vae_msgs_stacked, cache, actions_arr, has_control):
    """JIT'd forward-backward over the full chain via two ``lax.scan`` calls.

    All inputs are stacked along axis 0 (length T for messages, T-1 for
    actions). Returns four stacked Gaussian pytrees.
    """
    vae_etas = vae_msgs_stacked.eta   # (T, d)
    vae_lams = vae_msgs_stacked.lam   # (T, d, d)

    # --- Forward: alpha[0] -> ... -> alpha[T-1] ---
    def fwd_step(alpha_prev, x):
        v_eta, v_lam, u = x
        mx_eta = alpha_prev.eta + v_eta
        mx_lam = alpha_prev.lam + v_lam
        m_x = Gaussian(eta=mx_eta, lam=mx_lam)
        alpha_next = ct_forward(m_x, cache, u if has_control else None)
        return alpha_next, (m_x, alpha_next)

    fwd_xs = (vae_etas[:-1], vae_lams[:-1], actions_arr)
    _, (m_xs_stack, alphas_rest) = jax.lax.scan(fwd_step, prior_x, fwd_xs)
    alphas_stack = Gaussian(
        eta=jnp.concatenate([prior_x.eta[None], alphas_rest.eta]),
        lam=jnp.concatenate([prior_x.lam[None], alphas_rest.lam]),
    )

    # --- Backward: beta[T-1] = vague -> ... -> beta[0] ---
    vague_beta = Gaussian(
        eta=jnp.zeros_like(prior_x.eta),
        lam=jnp.zeros_like(prior_x.lam),
    )

    def bwd_step(beta_next, x):
        v_eta, v_lam, u = x
        my_eta = beta_next.eta + v_eta
        my_lam = beta_next.lam + v_lam
        m_y = Gaussian(eta=my_eta, lam=my_lam)
        beta_prev = ct_backward(m_y, cache, u if has_control else None)
        return beta_prev, (m_y, beta_prev)

    bwd_xs = (vae_etas[1:], vae_lams[1:], actions_arr)
    _, (m_ys_stack, betas_prev) = jax.lax.scan(
        bwd_step, vague_beta, bwd_xs, reverse=True
    )
    betas_stack = Gaussian(
        eta=jnp.concatenate([betas_prev.eta, vague_beta.eta[None]]),
        lam=jnp.concatenate([betas_prev.lam, vague_beta.lam[None]]),
    )

    return alphas_stack, betas_stack, m_xs_stack, m_ys_stack


def forward_backward(prior_x, vae_msgs, cache, actions=None):
    """Run forward-backward belief propagation on a latent chain.

    Parameters
    ----------
    prior_x : Gaussian — prior on x[0]
    vae_msgs : list of Gaussian, length T (or already-stacked Gaussian)
    cache : CTCache — transition node cache
    actions : list, length T-1, or stacked array (T-1, du), optional.

    Returns
    -------
    alphas, betas : Gaussian pytrees with leading axis T
    m_xs, m_ys    : Gaussian pytrees with leading axis T-1
    """
    if isinstance(vae_msgs, list):
        vae_stack = _stack_gaussians(vae_msgs)
        T = len(vae_msgs)
    else:
        vae_stack = vae_msgs
        T = vae_stack.eta.shape[0]

    has_control = (actions is not None) and (cache.mB is not None)
    if has_control:
        actions_arr = jnp.stack(actions) if isinstance(actions, list) else actions
    else:
        # Placeholder — body skips B·u when has_control=False (static).
        actions_arr = jnp.zeros((T - 1, 1))

    return _fb_scan(prior_x, vae_stack, cache, actions_arr, has_control)


# ---------------------------------------------------------------------------
# VMP message accumulation via vmap
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("has_control",))
def _accumulate_scan(m_xs_stack, m_ys_stack, cache, actions_arr,
                     prior_a, prior_W, prior_b, has_control):
    """vmap'd per-transition VMP messages, summed into the prior."""

    def per_step(m_x, m_y, u):
        u_arg = u if has_control else None
        q_yx = ct_marginal_yx(m_y, m_x, cache, u_arg)
        msg_a = ct_message_a(q_yx, cache, u_arg)
        delta = ct_message_W(q_yx, cache, u_arg)
        if has_control:
            msg_b = ct_message_b(q_yx, cache, u)
        else:
            msg_b = Gaussian(
                eta=jnp.zeros_like(prior_b.eta) if prior_b is not None else jnp.zeros(0),
                lam=jnp.zeros_like(prior_b.lam) if prior_b is not None else jnp.zeros((0, 0)),
            )
        return msg_a, delta, msg_b

    msgs_a, deltas, msgs_b = jax.vmap(per_step)(m_xs_stack, m_ys_stack, actions_arr)
    n_ct = m_xs_stack.eta.shape[0]

    q_a = Gaussian(
        eta=prior_a.eta + msgs_a.eta.sum(axis=0),
        lam=prior_a.lam + msgs_a.lam.sum(axis=0),
    )
    q_W = Wishart(df=prior_W.df + n_ct, inv_scale=prior_W.inv_scale + deltas.sum(axis=0))
    if has_control:
        q_b = Gaussian(
            eta=prior_b.eta + msgs_b.eta.sum(axis=0),
            lam=prior_b.lam + msgs_b.lam.sum(axis=0),
        )
    else:
        q_b = None
    return q_a, q_W, q_b


def accumulate_vmp_messages(m_xs, m_ys, cache, actions,
                            prior_a, prior_W, prior_b=None):
    """Accumulate VMP messages for transition parameters from BP marginals.

    ``m_xs``/``m_ys`` may be stacked Gaussian pytrees (from
    :func:`forward_backward`) or Python lists thereof.
    """
    if isinstance(m_xs, list):
        m_xs = _stack_gaussians(m_xs)
    if isinstance(m_ys, list):
        m_ys = _stack_gaussians(m_ys)

    n_ct = m_xs.eta.shape[0]
    has_control = (prior_b is not None) and (cache.mB is not None)
    if has_control:
        actions_arr = jnp.stack(actions) if isinstance(actions, list) else actions
    else:
        actions_arr = jnp.zeros((n_ct, 1))

    return _accumulate_scan(
        m_xs, m_ys, cache, actions_arr, prior_a, prior_W, prior_b, has_control,
    )


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

@jax.jit
def _marginals_vmap(alphas_stack, betas_stack, vae_stack):
    return jax.vmap(combine_gaussians)(alphas_stack, betas_stack, vae_stack)


def compute_marginals(alphas, betas, vae_msgs):
    """Compute marginal posteriors q(x[t]) from BP messages.

    Inputs may be stacked Gaussians or lists thereof.
    """
    if isinstance(alphas, list):
        alphas = _stack_gaussians(alphas)
    if isinstance(betas, list):
        betas = _stack_gaussians(betas)
    if isinstance(vae_msgs, list):
        vae_msgs = _stack_gaussians(vae_msgs)
    return _marginals_vmap(alphas, betas, vae_msgs)
