"""Belief propagation and VMP utilities for latent chain models.

The full T-length sequence is stacked into a single pytree so forward-backward,
VMP accumulation, and the planning loop each run inside one JIT'd graph.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .config import PLAN_VMP_ITERATIONS
from .distributions import Gaussian, Wishart, combine_gaussians, gaussian_mean
from .nodes.transition import (
    CTCache,
    CTMeta,
    ct_backward,
    ct_forward,
    ct_marginal_yx,
    ct_message_a,
    ct_message_b,
    ct_message_u,
    ct_message_W,
)


class ActionPosterior(NamedTuple):
    """Exact Gaussian action posterior returned for diagnostics.

    Mean and marginal_covariance are indexed by planning step.
    joint_covariance retains cross-time action correlations.
    """
    mean: jnp.ndarray
    marginal_covariance: jnp.ndarray
    joint_covariance: jnp.ndarray


def _stack(msgs):
    return Gaussian(
        eta=jnp.stack([m.eta for m in msgs]),
        lam=jnp.stack([m.lam for m in msgs]),
    )


def _as_stack(x):
    return _stack(x) if isinstance(x, list) else x


# ---------------------------------------------------------------------------
# Forward-backward
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("has_control",))
def _fb_scan(prior_x, obs, cache, actions, has_control):
    def fwd(alpha_prev, x):
        e, l, u = x
        m_x = Gaussian(eta=alpha_prev.eta + e, lam=alpha_prev.lam + l)
        alpha = ct_forward(m_x, cache, u if has_control else None)
        return alpha, (m_x, alpha)

    _, (m_xs, alphas_rest) = jax.lax.scan(
        fwd, prior_x, (obs.eta[:-1], obs.lam[:-1], actions))
    alphas = Gaussian(
        eta=jnp.concatenate([prior_x.eta[None], alphas_rest.eta]),
        lam=jnp.concatenate([prior_x.lam[None], alphas_rest.lam]),
    )

    vague = Gaussian(eta=jnp.zeros_like(prior_x.eta), lam=jnp.zeros_like(prior_x.lam))

    def bwd(beta_next, x):
        e, l, u = x
        m_y = Gaussian(eta=beta_next.eta + e, lam=beta_next.lam + l)
        beta = ct_backward(m_y, cache, u if has_control else None)
        return beta, (m_y, beta)

    _, (m_ys, betas_prev) = jax.lax.scan(
        bwd, vague, (obs.eta[1:], obs.lam[1:], actions), reverse=True)
    betas = Gaussian(
        eta=jnp.concatenate([betas_prev.eta, vague.eta[None]]),
        lam=jnp.concatenate([betas_prev.lam, vague.lam[None]]),
    )
    return alphas, betas, m_xs, m_ys


def forward_backward(prior_x, obs_msgs, cache, actions=None):
    """Forward-backward on a chain ``x[t] → x[t+1]`` with leaf messages ``obs_msgs``.

    ``actions`` is None or a length-(T-1) list / stacked (T-1, du) array.
    Returns ``(alphas, betas, m_xs, m_ys)`` — stacked Gaussians of length T, T,
    T-1, T-1.
    """
    obs = _as_stack(obs_msgs)
    T = obs.eta.shape[0]
    if actions is not None and cache.mB is None:
        raise ValueError(
            "forward_backward: `actions` were provided but the model has no control matrix "
            "(cache.mB is None) — supply a model with B, or drop `actions`.")
    has_control = actions is not None and cache.mB is not None
    if has_control:
        actions = jnp.stack(actions) if isinstance(actions, list) else actions
    else:
        actions = jnp.zeros((T - 1, 1))   # placeholder; body skips B·u
    return _fb_scan(prior_x, obs, cache, actions, has_control)


# ---------------------------------------------------------------------------
# VMP accumulation for q(A), q(B), q(W)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("has_control",))
def _accumulate(m_xs, m_ys, cache, actions, prior_a, prior_W, prior_b, has_control):
    def per_step(m_x, m_y, u):
        u_arg = u if has_control else None
        q_yx = ct_marginal_yx(m_y, m_x, cache, u_arg)
        msg_a = ct_message_a(q_yx, cache, u_arg)
        delta = ct_message_W(q_yx, cache, u_arg)
        msg_b = (ct_message_b(q_yx, cache, u) if has_control
                 else Gaussian(eta=jnp.zeros_like(prior_b.eta) if prior_b is not None else jnp.zeros(0),
                               lam=jnp.zeros_like(prior_b.lam) if prior_b is not None else jnp.zeros((0, 0))))
        return msg_a, delta, msg_b

    msgs_a, deltas, msgs_b = jax.vmap(per_step)(m_xs, m_ys, actions)
    n = m_xs.eta.shape[0]
    q_a = Gaussian(eta=prior_a.eta + msgs_a.eta.sum(0),
                   lam=prior_a.lam + msgs_a.lam.sum(0))
    q_W = Wishart(df=prior_W.df + n, inv_scale=prior_W.inv_scale + deltas.sum(0))
    q_b = (Gaussian(eta=prior_b.eta + msgs_b.eta.sum(0),
                    lam=prior_b.lam + msgs_b.lam.sum(0))
           if has_control else None)
    return q_a, q_W, q_b


def accumulate_vmp_messages(m_xs, m_ys, cache, actions, prior_a, prior_W, prior_b=None):
    """Accumulate VMP messages into q(A), q(B), q(W) from BP marginals."""
    m_xs, m_ys = _as_stack(m_xs), _as_stack(m_ys)
    n = m_xs.eta.shape[0]
    has_control = prior_b is not None and cache.mB is not None
    if has_control:
        if actions is None:
            raise ValueError(
                "accumulate_vmp_messages: control is enabled (prior_b and cache.mB set) "
                "but `actions` is None — provide the per-transition controls.")
        actions = jnp.stack(actions) if isinstance(actions, list) else actions
    else:
        actions = jnp.zeros((n, 1))
    return _accumulate(m_xs, m_ys, cache, actions, prior_a, prior_W, prior_b, has_control)


@partial(jax.jit, static_argnames=("n_iterations",))
def fit_vmp_grouped_controlled(prior_x, obs_groups, action_groups,
                               prior_a, prior_W, prior_b,
                               q_a, q_W, q_b, n_iterations):
    """Fused joint VMP for trajectory groups padded by equal length.

    Each group is a stacked Gaussian ``(batch, time, dim)`` plus controls of
    shape ``(batch, time-1, du)``. Forward/backward remains independent per
    trajectory; only conjugate sufficient statistics are pooled, exactly as
    in ``LearnedLinear(mode="joint")``.
    """
    d = prior_x.eta.shape[0]
    meta = CTMeta(lambda a: a.reshape(d, d))

    def iteration(_, qs):
        qa, qW, qb = qs
        cache = CTCache(qa, qW, meta, qb)
        all_mx_eta, all_mx_lam = [], []
        all_my_eta, all_my_lam = [], []
        all_actions = []
        for obs, actions in zip(obs_groups, action_groups):
            def bp(eta, lam, u):
                _, _, mx, my = _fb_scan(
                    prior_x, Gaussian(eta, lam), cache, u, True)
                return mx, my
            mx, my = jax.vmap(bp)(obs.eta, obs.lam, actions)
            all_mx_eta.append(mx.eta.reshape((-1, d)))
            all_mx_lam.append(mx.lam.reshape((-1, d, d)))
            all_my_eta.append(my.eta.reshape((-1, d)))
            all_my_lam.append(my.lam.reshape((-1, d, d)))
            all_actions.append(actions.reshape((-1, actions.shape[-1])))
        mx = Gaussian(jnp.concatenate(all_mx_eta), jnp.concatenate(all_mx_lam))
        my = Gaussian(jnp.concatenate(all_my_eta), jnp.concatenate(all_my_lam))
        actions = jnp.concatenate(all_actions)
        return _accumulate(
            mx, my, cache, actions, prior_a, prior_W, prior_b, True)

    return jax.lax.fori_loop(0, n_iterations, iteration, (q_a, q_W, q_b))


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------

@jax.jit
def _marginals_vmap(alphas, betas, obs):
    return jax.vmap(combine_gaussians)(alphas, betas, obs)


def compute_marginals(alphas, betas, obs_msgs):
    """Smoothed marginal q(x[t]) = α[t] · β[t] · obs[t] (in information form)."""
    return _marginals_vmap(_as_stack(alphas), _as_stack(betas), _as_stack(obs_msgs))


# ---------------------------------------------------------------------------
# Planning as inference — VMP on the action sequence
# ---------------------------------------------------------------------------

def infer_actions(prior_x, obs_msgs, cache, prior_u,
                  n_iterations=PLAN_VMP_ITERATIONS, verbose=False):
    """Infer ``u[0..T-2]`` by VMP on the action sequence: alternate
    forward-backward through the chain (with current action means) with
    per-transition action posterior updates from the CT factor.
    """
    T = len(obs_msgs) if isinstance(obs_msgs, list) else obs_msgs.eta.shape[0]
    du = prior_u.eta.shape[0]
    q_us = Gaussian(
        eta=jnp.broadcast_to(prior_u.eta, (T - 1, du)),
        lam=jnp.broadcast_to(prior_u.lam, (T - 1, du, du)),
    )

    def update(m_y, m_x, u):
        return combine_gaussians(prior_u, ct_message_u(ct_marginal_yx(m_y, m_x, cache, u), cache))
    vupd = jax.vmap(update)

    pbar = tqdm(range(n_iterations), desc="plan", disable=not verbose)
    for _ in pbar:
        u_means = jax.vmap(gaussian_mean)(q_us)
        _, _, m_xs, m_ys = forward_backward(prior_x, obs_msgs, cache, u_means)
        q_us = vupd(m_ys, m_xs, u_means)
    return jax.vmap(gaussian_mean)(q_us)


def infer_actions_exact_numpy(prior_x, obs_msgs, cache, prior_u):
    """Planning as inference, exactly — one forward–backward sweep.

    :func:`infer_actions` runs coordinate-ascent VMP with a mean-field cut
    between the state chain and the actions, which under-disperses on strongly
    coupled action sequences. Removing the cut makes planning *exact*: group
    each time slice as ``s[t] = [x[t]; u[t]]`` and the joint is a Gaussian
    chain, so sum-product — the same α (forward elimination) and β (back
    substitution) recursions as Kalman smoothing, here in information form —
    yields the action posterior in ONE sweep. This is LQG duality: the
    smoothing sweep *is* the optimal controller.

    Runs in float64 at Python level — the belief/goal/transition precision
    scales span more than float32 resolves; a numerics decision, not style.
    Uses the posterior-mean transition ``(E[A], E[B], E[W])`` from the cache.
    (Raw expected-moment corrections are deliberately NOT applied: the
    ``xᵀ·contraction(Va)·x`` free-energy term is a state-magnitude penalty
    that drags plans toward the origin — correct for smoothing, pathological
    as a planning cost on non-centred latents.) Returns the ``(T-1, du)``
    posterior-mean actions.
    """
    d, du = cache.dx, cache.du
    ds = d + du
    obs_list = obs_msgs if isinstance(obs_msgs, list) else [
        Gaussian(eta=obs_msgs.eta[t], lam=obs_msgs.lam[t])
        for t in range(obs_msgs.eta.shape[0])]
    T = len(obs_list)
    f64 = lambda a: np.asarray(a, dtype=np.float64)
    A, B, W = f64(cache.mA), f64(cache.mB), f64(cache.mW)
    AtW, BtW = A.T @ W, B.T @ W

    # per-slice information: D[t] (ds×ds) on s[t], coupling C[t] to s[t+1]
    D = np.zeros((T, ds, ds)); e = np.zeros((T, ds))
    C = np.zeros((T - 1, ds, ds))
    for t, ob in enumerate(obs_list):
        D[t, :d, :d] += f64(ob.lam); e[t, :d] += f64(ob.eta)
        D[t, d:, d:] += f64(prior_u.lam); e[t, d:] += f64(prior_u.eta)
    D[0, :d, :d] += f64(prior_x.lam); e[0, :d] += f64(prior_x.eta)
    for t in range(T - 1):
        # transition factor: quadratic of  x[t+1] − A x[t] − B u[t]  in W
        D[t, :d, :d] += AtW @ A
        D[t, d:, d:] += BtW @ B
        D[t, :d, d:] += AtW @ B
        D[t, d:, :d] += (AtW @ B).T
        D[t + 1, :d, :d] += W
        C[t, :d, :d] = -AtW          # rows s[t], cols s[t+1] (x-part only)
        C[t, d:, :d] = -BtW

    # α sweep: forward elimination of the block-tridiagonal information form
    Dt = D.copy(); et = e.copy()
    G = np.zeros_like(C)
    for t in range(T - 1):
        G[t] = np.linalg.solve(Dt[t], C[t])
        Dt[t + 1] -= C[t].T @ G[t]
        et[t + 1] -= C[t].T @ np.linalg.solve(Dt[t], et[t])
    # β sweep: back substitution → posterior means of every slice
    mu = np.zeros((T, ds))
    mu[T - 1] = np.linalg.solve(Dt[T - 1], et[T - 1])
    for t in range(T - 2, -1, -1):
        mu[t] = np.linalg.solve(Dt[t], et[t] - C[t] @ mu[t + 1])
    with jax.enable_x64(True):
        return jnp.asarray(mu[:-1, d:])          # u[0 .. T-2]


@jax.jit
def _infer_actions_exact_jax(prior_eta, prior_lam, obs_eta, obs_lam,
                             A, B, W, prior_u_eta, prior_u_lam):
    """Device-resident block-tridiagonal exact-planning sweep."""
    T, d = obs_eta.shape
    du = prior_u_eta.shape[0]
    ds = d + du

    AtW, BtW = A.T @ W, B.T @ W
    AtWA, BtWB, AtWB = AtW @ A, BtW @ B, AtW @ B

    D = jnp.zeros((T, ds, ds), dtype=jnp.float64)
    e = jnp.zeros((T, ds), dtype=jnp.float64)
    C = jnp.zeros((T - 1, ds, ds), dtype=jnp.float64)

    D = D.at[:, :d, :d].add(obs_lam)
    e = e.at[:, :d].add(obs_eta)
    D = D.at[:, d:, d:].add(prior_u_lam)
    e = e.at[:, d:].add(prior_u_eta)
    D = D.at[0, :d, :d].add(prior_lam)
    e = e.at[0, :d].add(prior_eta)

    D = D.at[:-1, :d, :d].add(AtWA)
    D = D.at[:-1, d:, d:].add(BtWB)
    D = D.at[:-1, :d, d:].add(AtWB)
    D = D.at[:-1, d:, :d].add(AtWB.T)
    D = D.at[1:, :d, :d].add(W)
    C = C.at[:, :d, :d].set(-AtW)
    C = C.at[:, d:, :d].set(-BtW)

    def eliminate(carry, x):
        Dt, et = carry
        Dnext, enext, Ct = x
        solved_C = jnp.linalg.solve(Dt, Ct)
        solved_e = jnp.linalg.solve(Dt, et)
        Dnext = Dnext - Ct.T @ solved_C
        enext = enext - Ct.T @ solved_e
        return (Dnext, enext), (Dnext, enext)

    (_, _), (Drest, erest) = jax.lax.scan(
        eliminate, (D[0], e[0]), (D[1:], e[1:], C))
    Dt = jnp.concatenate((D[:1], Drest), axis=0)
    et = jnp.concatenate((e[:1], erest), axis=0)

    mu_last = jnp.linalg.solve(Dt[-1], et[-1])

    def substitute(mu_next, x):
        Dt_t, et_t, Ct = x
        mu = jnp.linalg.solve(Dt_t, et_t - Ct @ mu_next)
        return mu, mu

    _, mu_prev = jax.lax.scan(
        substitute, mu_last, (Dt[:-1], et[:-1], C), reverse=True)
    mu = jnp.concatenate((mu_prev, mu_last[None]), axis=0)
    return mu[:-1, d:]


def infer_actions_exact(prior_x, obs_msgs, cache, prior_u):
    """Exact planning-as-inference on the active JAX device.

    The equations are identical to :func:`infer_actions_exact_numpy`, retained
    as an independent CPU reference.  The complete float64 alpha/beta sweep is
    compiled and stays device-resident.
    """
    obs = _as_stack(obs_msgs)
    # Exact planning needs float64, while the VAE should remain float32.
    with jax.enable_x64(True):
        f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
        return _infer_actions_exact_jax(
            f64(prior_x.eta), f64(prior_x.lam),
            f64(obs.eta), f64(obs.lam),
            f64(cache.mA), f64(cache.mB), f64(cache.mW),
            f64(prior_u.eta), f64(prior_u.lam))


@jax.jit
def _infer_actions_exact_posterior_jax(
        prior_eta, prior_lam, obs_eta, obs_lam,
        A, B, W, prior_u_eta, prior_u_lam):
    """Dense diagnostic solve of the same exact Gaussian planning graph."""
    T, d = obs_eta.shape
    du = prior_u_eta.shape[0]
    ds = d + du
    AtW, BtW = A.T @ W, B.T @ W

    D = jnp.zeros((T, ds, ds), dtype=jnp.float64)
    e = jnp.zeros((T, ds), dtype=jnp.float64)
    C = jnp.zeros((T - 1, ds, ds), dtype=jnp.float64)
    D = D.at[:, :d, :d].add(obs_lam)
    e = e.at[:, :d].add(obs_eta)
    D = D.at[:, d:, d:].add(prior_u_lam)
    e = e.at[:, d:].add(prior_u_eta)
    D = D.at[0, :d, :d].add(prior_lam)
    e = e.at[0, :d].add(prior_eta)
    D = D.at[:-1, :d, :d].add(AtW @ A)
    D = D.at[:-1, d:, d:].add(BtW @ B)
    D = D.at[:-1, :d, d:].add(AtW @ B)
    D = D.at[:-1, d:, :d].add((AtW @ B).T)
    D = D.at[1:, :d, :d].add(W)
    C = C.at[:, :d, :d].set(-AtW)
    C = C.at[:, d:, :d].set(-BtW)

    precision = jnp.zeros((T * ds, T * ds), dtype=jnp.float64)
    for t in range(T):
        start = t * ds
        precision = precision.at[
            start:start + ds, start:start + ds].add(D[t])
    for t in range(T - 1):
        left, right = t * ds, (t + 1) * ds
        precision = precision.at[
            left:left + ds, right:right + ds].add(C[t])
        precision = precision.at[
            right:right + ds, left:left + ds].add(C[t].T)

    covariance = jnp.linalg.solve(
        precision, jnp.eye(T * ds, dtype=jnp.float64))
    mean = covariance @ e.reshape(-1)
    action_index = (
        jnp.arange(T - 1)[:, None] * ds + d + jnp.arange(du)[None]).reshape(-1)
    action_mean = mean[action_index].reshape((T - 1, du))
    joint_covariance = covariance[action_index[:, None], action_index[None, :]]
    joint_blocks = joint_covariance.reshape((T - 1, du, T - 1, du))
    marginal_covariance = jnp.stack([
        joint_blocks[t, :, t, :] for t in range(T - 1)])
    return ActionPosterior(
        action_mean, marginal_covariance, joint_covariance)


def infer_actions_exact_posterior(prior_x, obs_msgs, cache, prior_u):
    """Return means and covariances from the exact action posterior.

    This is intended for uncertainty telemetry and visualization. The action
    means are identical to infer_actions_exact; the ordinary planner keeps its
    faster block-tridiagonal mean-only sweep.
    """
    obs = _as_stack(obs_msgs)
    with jax.enable_x64(True):
        f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
        return _infer_actions_exact_posterior_jax(
            f64(prior_x.eta), f64(prior_x.lam),
            f64(obs.eta), f64(obs.lam),
            f64(cache.mA), f64(cache.mB), f64(cache.mW),
            f64(prior_u.eta), f64(prior_u.lam))
