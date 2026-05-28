"""Inference engine: forward-backward BP on latent chain + VMP for shared params.

Combines belief propagation (messages on latent states) with variational
message passing (updates for transition matrix and precision).

Supports optional control inputs u[t] with learned control matrix B.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .distributions import (
    Gaussian, Wishart,
    gaussian_mean, gaussian_mean_cov, gaussian_prior,
    vague_gaussian, wishart_mean,
)
from .defaults import (
    PRIOR_W_DF, PRIOR_A_COV, PRIOR_B_COV, INIT_A_COV, INIT_B_COV,
)
from .nodes.transition import CTMeta, CTCache
from .nodes.observation import vae_predict
from .message_passing import (
    encode_observations, forward_backward,
    accumulate_vmp_messages, compute_marginals, infer_actions,
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
    vae,
    *,
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
        xₜ ~ N(A·xₜ₋₁ + B·uₜ₋₁, W⁻¹)  — A = reshape(a, (d, d))
        yₜ ~ VAENode(xₜ)

    Parameters
    ----------
    observations : list of arrays or None
        Observed images; ``None`` entries are missing (unobserved).
    vae : :class:`jopa.nn.vae.VAEAdapter`
        Bundle of jitted ``encode``/``decode`` closures plus ``latent_dim``,
        as returned by :func:`jopa.nn.vae.make_encode_decode`. ``encode``
        maps an image to ``(mean, log_std)``; ``decode`` maps a latent ``z``
        back to an image.
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
    encode_fn, decode_fn, d = vae.encode, vae.decode, vae.latent_dim
    da = d * d
    T_obs = len(observations)
    meta = CTMeta(lambda a: a.reshape(d, d))

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
    marginals = compute_marginals(alphas, betas, all_vae)  # stacked Gaussian

    lat_means, lat_covs = jax.vmap(gaussian_mean_cov)(marginals)
    preds = list(jax.vmap(lambda q: vae_predict(q, decode_fn))(marginals))

    ma = gaussian_mean(q_a)
    mW = wishart_mean(q_W)
    control_matrix = None
    if has_control:
        mb = gaussian_mean(q_b)
        control_matrix = mb.reshape(d, du)

    return InferenceResult(
        latent_means=lat_means,
        latent_covs=lat_covs,
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
    actions: jnp.ndarray               # (n_ct, du) inferred action sequence
    latent_means: jnp.ndarray          # (T, d) planned trajectory
    predictions: list                  # decoded images for each step


def plan(
    observations: Sequence[jnp.ndarray | None],
    vae,
    q_a: Gaussian,
    q_W: Wishart,
    q_b: Gaussian,
    action_dim: int,
    *,
    prior_x: Gaussian | None = None,
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
    prior_x : Gaussian, optional
        Belief over the initial state x₀. Defaults to N(0, I). Pass a carried
        filter belief here for online receding-horizon control — the current
        state (incl. velocity-like latent dims) then comes from the belief
        rather than a single re-encoded frame.

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
    encode_fn, decode_fn, d = vae.encode, vae.decode, vae.latent_dim
    du = action_dim
    meta = CTMeta(lambda a: a.reshape(d, d))

    vae_msgs = encode_observations(observations, encode_fn, d)

    cache = CTCache(q_a, q_W, meta, q_b)
    if prior_x is None:
        prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))
        start_mean = None
    else:
        start_mean = gaussian_mean(prior_x)

    # Action prior derived from the start→goal latent shift — see docstring.
    # Start anchor is the carried belief mean (if given) else the first
    # observed frame; goal anchor is the last observed frame. Falls back to
    # an identity-scale prior on B·u if no shift can be measured.
    mB = gaussian_mean(q_b).reshape(d, du)
    b_col_var = jnp.sum(mB ** 2, axis=0) + 1e-8

    present = [i for i, o in enumerate(observations) if o is not None]
    mu_start, mu_end = None, None
    if start_mean is not None and present:
        mu_start, mu_end = start_mean, gaussian_mean(vae_msgs[present[-1]])
    elif len(present) >= 2:
        mu_start, mu_end = gaussian_mean(vae_msgs[present[0]]), gaussian_mean(vae_msgs[present[-1]])
    if mu_start is not None:
        latent_shift_sq = jnp.maximum(jnp.sum((mu_end - mu_start) ** 2), 1e-8)
    else:
        latent_shift_sq = jnp.array(1.0)

    prior_u_lam = jnp.diag(b_col_var / latent_shift_sq)
    prior_u = Gaussian(eta=jnp.zeros(du), lam=prior_u_lam)

    u_means = infer_actions(prior_x, vae_msgs, cache, prior_u,
                            n_iterations=n_iterations, verbose=verbose)

    # Final marginals with converged actions
    alphas, betas, _, _ = forward_backward(prior_x, vae_msgs, cache, u_means)
    marginals = compute_marginals(alphas, betas, vae_msgs)
    lat_means = jax.vmap(gaussian_mean)(marginals)
    preds = list(jax.vmap(lambda q: vae_predict(q, decode_fn))(marginals))

    return PlanResult(
        actions=u_means,
        latent_means=lat_means,
        predictions=preds,
    )
