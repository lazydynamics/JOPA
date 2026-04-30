"""Inference engine: forward-backward BP on latent chain + VMP for shared params.

Combines belief propagation (messages on latent states) with variational
message passing (updates for transition matrix and precision).

Supports optional control inputs u[t] with learned control matrix B.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

import jax.numpy as jnp
from tqdm import tqdm

from .distributions import (
    Gaussian, Wishart,
    combine_gaussians, gaussian_mean, gaussian_mean_cov, gaussian_prior,
    vague_gaussian, wishart_mean,
)
from .defaults import (
    PRIOR_W_DF, PRIOR_A_COV, PRIOR_B_COV, INIT_A_COV, INIT_B_COV,
)
from .nodes.transition import (
    CTMeta, CTCache,
    ct_marginal_yx, ct_message_u,
)
from .nodes.observation import vae_predict
from .message_passing import (
    encode_observations, forward_backward,
    accumulate_vmp_messages, compute_marginals,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Holds inference outputs."""
    latent_means: jnp.ndarray          # (T, d) posterior means of x[t]
    latent_covs: jnp.ndarray           # (T, d, d) posterior covariances
    transition_matrix: jnp.ndarray     # (dy, dx) learned A = reshape(E[a])
    transition_precision: jnp.ndarray  # (dy, dy) E[W]
    predictions: list                  # decoded images for every time step
    q_a: Gaussian = field(repr=False)
    q_W: Wishart = field(repr=False)
    control_matrix: jnp.ndarray | None = None   # (dy, du) learned B
    q_b: Gaussian | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def infer(
    observations: Sequence[jnp.ndarray | None],
    encode_fn: Callable,
    decode_fn: Callable,
    latent_dim: int,
    *,
    transform_fn: Callable | None = None,
    actions: Sequence[jnp.ndarray] | None = None,
    action_dim: int | None = None,
    n_predict: int = 0,
    predict_actions: Sequence[jnp.ndarray] | None = None,
    n_iterations: int = 50,
    prior_W_df: float = PRIOR_W_DF,
    prior_a_cov: float = PRIOR_A_COV,
    prior_b_cov: float = PRIOR_B_COV,
    init_a_cov: float = INIT_A_COV,
    init_b_cov: float = INIT_B_COV,
    init_a_mean: jnp.ndarray | None = None,
    prior_a_mean: jnp.ndarray | None = None,
    verbose: bool = True,
) -> InferenceResult:
    """Run VMP + BP inference on a VAE state-space model.

    Model
    -----
        W  ~ Wishart(prior_W_df, I)
        a  ~ N(0, prior_a_cov · I)        — vec(transition matrix A)
        b  ~ N(0, prior_b_cov · I)        — vec(control matrix B)
        x₁ ~ N(0, I)
        xₜ ~ N(A·xₜ₋₁ + B·uₜ₋₁, W⁻¹)  — A = transform_fn(a)
        yₜ ~ VAENode(xₜ)

    Parameters
    ----------
    observations : list of arrays or None
        Observed images; ``None`` entries are missing (unobserved).
    encode_fn : (image) → (mean, log_std)
    decode_fn : (z) → image
    transform_fn : (a) → A
    latent_dim : int
    actions : list of (du,) arrays, optional
        Control inputs u[t] for t=0..T_obs-2 (one per transition).
        Length must be T_obs - 1.
    action_dim : int, optional
        Dimensionality of actions. Required if actions is provided.
    n_predict : int
        Extra future steps to predict beyond observations.
    predict_actions : list of (du,) arrays, optional
        Actions for prediction steps. Length must be n_predict.
        If None and actions given, uses zeros.
    """
    d = latent_dim
    da = d * d
    T_obs = len(observations)
    T = T_obs + n_predict
    if transform_fn is None:
        transform_fn = lambda a: a.reshape(d, d)
    meta = CTMeta(transform_fn)

    has_control = actions is not None
    if has_control:
        assert action_dim is not None, "action_dim required when actions given"
        assert len(actions) == T_obs - 1, \
            f"Need {T_obs-1} actions (one per transition), got {len(actions)}"
        du = action_dim
        db = d * du

    # --- Precompute VAE messages (fixed across iterations) -----------------
    vae_msgs = encode_observations(observations, encode_fn, d)

    # --- Priors & initial marginals ----------------------------------------
    prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))
    prior_a = gaussian_prior(da, prior_a_cov, prior_a_mean)
    q_a = gaussian_prior(da, init_a_cov, init_a_mean)
    prior_W = Wishart(df=prior_W_df, inv_scale=jnp.eye(d))
    q_W = prior_W

    q_b, prior_b = None, None
    if has_control:
        prior_b = gaussian_prior(db, prior_b_cov)
        q_b = gaussian_prior(db, init_b_cov)

    obs_actions = list(actions) if has_control else None

    # --- VMP loop (observed portion only) ----------------------------------
    pbar = tqdm(range(n_iterations), desc="VMP", disable=not verbose)
    for _ in pbar:
        cache = CTCache(q_a, q_W, meta, q_b)
        alphas, betas, m_xs, m_ys = forward_backward(
            prior_x, vae_msgs, cache, obs_actions)
        q_a, q_W, q_b = accumulate_vmp_messages(
            m_xs, m_ys, cache, obs_actions, prior_a, prior_W, prior_b)

        ma = gaussian_mean(q_a)
        mA = meta.f(ma)
        postfix = {"det(A)": f"{jnp.linalg.det(mA):.4f}", "df(W)": f"{q_W.df:.0f}"}
        if has_control:
            mb = gaussian_mean(q_b)
            postfix["|B|"] = f"{jnp.linalg.norm(mb.reshape(d, du)):.4f}"
        pbar.set_postfix(postfix)

    # --- Final marginals (observed + predicted) ----------------------------
    cache = CTCache(q_a, q_W, meta, q_b)

    if has_control:
        all_actions = list(actions) + list(
            predict_actions if predict_actions is not None
            else [jnp.zeros(du)] * n_predict)
    else:
        all_actions = None

    all_vae = vae_msgs + [vague_gaussian(d)] * n_predict
    alphas, betas, _, _ = forward_backward(prior_x, all_vae, cache, all_actions)
    marginals = compute_marginals(alphas, betas, all_vae)

    lat_means, lat_covs = [], []
    for q in marginals:
        mu, cov = gaussian_mean_cov(q)
        lat_means.append(mu)
        lat_covs.append(cov)

    preds = [vae_predict(q, decode_fn) for q in marginals]

    ma = gaussian_mean(q_a)
    mW = wishart_mean(q_W)
    control_matrix = None
    if has_control:
        mb = gaussian_mean(q_b)
        control_matrix = mb.reshape(d, du)

    return InferenceResult(
        latent_means=jnp.stack(lat_means),
        latent_covs=jnp.stack(lat_covs),
        transition_matrix=meta.f(ma),
        transition_precision=mW,
        predictions=preds,
        q_a=q_a,
        q_W=q_W,
        control_matrix=control_matrix,
        q_b=q_b,
    )


# ---------------------------------------------------------------------------
# Planning as inference: infer actions to reach a goal
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Holds planning outputs."""
    actions: list[jnp.ndarray]         # inferred action sequence
    latent_means: jnp.ndarray          # (T, d) planned trajectory
    predictions: list                  # decoded images for each step


def plan(
    observations: Sequence[jnp.ndarray | None],
    encode_fn: Callable,
    decode_fn: Callable,
    q_a: Gaussian,
    q_W: Wishart,
    q_b: Gaussian,
    latent_dim: int,
    action_dim: int,
    *,
    transform_fn: Callable | None = None,
    n_iterations: int = 50,
    verbose: bool = True,
) -> PlanResult:
    """Infer actions by conditioning on observed images (start + goal).

    Fix the model (A, B, W) and treat actions u[t] as latent variables.
    Observations can be provided at any subset of time steps (None = free).
    VMP+BP infers both latent states and actions jointly.

    Parameters
    ----------
    observations : list of arrays or None
        Images at each time step. None = no observation (action is free).
        Typically: [start_image, None, ..., None, goal_image].
    q_a, q_W, q_b : learned model parameters (fixed).

    Notes
    -----
    The prior on actions is derived from the task itself, with no tunable
    parameter. For each action component i:

        var(u_i) = ‖μ_goal − μ_start‖² / |B_col_i|²

    Reads as: "a priori, a single action can produce a latent shift up to
    the task's total start→goal distance." Permissive — the dynamics
    likelihood selects the actual action magnitudes within this envelope.
    Scale-invariant in u (|B|² absorbs units) and in latent space (uses
    observed distance, not an arbitrary radius).
    """
    d = latent_dim
    du = action_dim
    T = len(observations)
    if transform_fn is None:
        transform_fn = lambda a: a.reshape(d, d)
    meta = CTMeta(transform_fn)
    n_ct = T - 1

    vae_msgs = encode_observations(observations, encode_fn, d)

    cache = CTCache(q_a, q_W, meta, q_b)
    prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))

    # Action prior derived from the observed start→goal latent shift —
    # see docstring. Uses the first and last *observed* timesteps as
    # anchors; falls back to identity-scale prior on B·u if fewer than
    # two observations are provided.
    mB = gaussian_mean(q_b).reshape(d, du)
    b_col_var = jnp.sum(mB ** 2, axis=0) + 1e-8

    present = [i for i, o in enumerate(observations) if o is not None]
    if len(present) >= 2:
        mu_start = gaussian_mean(vae_msgs[present[0]])
        mu_end = gaussian_mean(vae_msgs[present[-1]])
        latent_shift_sq = jnp.sum((mu_end - mu_start) ** 2)
        latent_shift_sq = jnp.maximum(latent_shift_sq, 1e-8)
    else:
        latent_shift_sq = jnp.array(1.0)

    prior_u_lam = jnp.diag(b_col_var / latent_shift_sq)
    prior_u = Gaussian(eta=jnp.zeros(du), lam=prior_u_lam)

    # Initialise action posteriors at the prior
    q_us = [prior_u for _ in range(n_ct)]

    pbar = tqdm(range(n_iterations), desc="Plan", disable=not verbose)
    for _ in pbar:
        u_means = [gaussian_mean(qu) for qu in q_us]

        # BP: forward-backward with current action means
        alphas, betas, m_xs, m_ys = forward_backward(
            prior_x, vae_msgs, cache, u_means)

        # VMP: update each action posterior from its CT factor message + prior
        for k in range(n_ct):
            q_yx = ct_marginal_yx(m_ys[k], m_xs[k], cache, u_means[k])
            msg_u = ct_message_u(q_yx, cache)
            q_us[k] = combine_gaussians(prior_u, msg_u)

        u_norms = [float(jnp.linalg.norm(gaussian_mean(qu))) for qu in q_us]
        pbar.set_postfix({
            "mean|u|": f"{sum(u_norms)/len(u_norms):.4f}",
            "max|u|": f"{max(u_norms):.4f}",
        })

    # Final marginals with converged actions
    u_means = [gaussian_mean(qu) for qu in q_us]
    alphas, betas, _, _ = forward_backward(prior_x, vae_msgs, cache, u_means)
    marginals = compute_marginals(alphas, betas, vae_msgs)
    lat_means = [gaussian_mean(q) for q in marginals]
    preds = [vae_predict(q, decode_fn) for q in marginals]

    return PlanResult(
        actions=u_means,
        latent_means=jnp.stack(lat_means),
        predictions=preds,
    )
