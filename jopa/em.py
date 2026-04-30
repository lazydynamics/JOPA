"""Variational EM as message passing.

Frames Variational EM as a message-passing scheme on a factor graph where:
  E-step: VMP+BP messages on the latent chain → posterior marginals q(x[t])
  M-step: gradient-based messages at the VAE observation nodes → updated params

See Senoz et al. "Variational Message Passing and Local Constraint Manipulation
in Factor Graphs" for the message-passing interpretation of EM.

Supports multiple trajectories, each with its own action sequence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .distributions import (
    Gaussian, Wishart,
    gaussian_mean, gaussian_mean_cov, gaussian_prior,
    wishart_mean,
)
from .defaults import (
    PRIOR_W_DF, PRIOR_A_COV, PRIOR_B_COV, INIT_A_COV, INIT_B_COV,
)
from .nodes.transition import CTMeta, CTCache
from .message_passing import (
    encode_observations, forward_backward,
    accumulate_vmp_messages, compute_marginals,
)
from .nn.vae import make_encode_decode, LOG_STD_CLIP, PROB_CLIP


# ---------------------------------------------------------------------------
# KL divergence:  KL( N(mu1, diag(s1^2)) || N(mu2, Sigma2) )
# ---------------------------------------------------------------------------

def _kl_diag_vs_full(mu1, log_std1, mu2, cov2):
    d = mu1.shape[0]
    var1 = jnp.exp(2.0 * log_std1)
    prec2 = jnp.linalg.inv(cov2)
    tr_term = jnp.sum(jnp.diag(prec2) * var1)
    diff = mu2 - mu1
    quad_term = diff @ prec2 @ diff
    log_det_2 = jnp.linalg.slogdet(cov2)[1]
    log_det_1 = 2.0 * jnp.sum(log_std1)
    return 0.5 * (tr_term + quad_term - d + log_det_2 - log_det_1)


# ---------------------------------------------------------------------------
# M-step loss
# ---------------------------------------------------------------------------

def _m_step_loss(params, model, images, posterior_means, posterior_covs, z_rng, beta_recon):
    batch_size = images.shape[0]
    mu_enc, log_std_enc = model.apply(params, images, method=model.encode)
    log_std_enc = jnp.clip(log_std_enc, *LOG_STD_CLIP)
    eps = jax.random.normal(z_rng, mu_enc.shape)
    z = mu_enc + jnp.exp(log_std_enc) * eps
    recon = model.apply(params, z, method=model.decode)
    flat = images.reshape(batch_size, -1)
    p = jnp.clip(recon, *PROB_CLIP)
    bce = -jnp.sum(flat * jnp.log(p) + (1 - flat) * jnp.log(1 - p), axis=-1)
    kl = jax.vmap(_kl_diag_vs_full)(mu_enc, log_std_enc, posterior_means, posterior_covs)
    return jnp.mean(beta_recon * bce + kl)


# ---------------------------------------------------------------------------
# E-step: run VMP+BP on one trajectory
# ---------------------------------------------------------------------------

def _e_step_trajectory(
    observations, actions, encode_fn, latent_dim,
    q_a, q_W, q_b, prior_a, prior_W, prior_b,
    meta, n_vmp_iterations,
):
    """E-step on a single trajectory. Returns updated q_a, q_W, q_b and marginals."""
    d = latent_dim
    vae_msgs = encode_observations(observations, encode_fn, d)
    prior_x = Gaussian(eta=jnp.zeros(d), lam=jnp.eye(d))

    for _ in range(n_vmp_iterations):
        cache = CTCache(q_a, q_W, meta, q_b)
        alphas, betas, m_xs, m_ys = forward_backward(
            prior_x, vae_msgs, cache, actions)
        q_a, q_W, q_b = accumulate_vmp_messages(
            m_xs, m_ys, cache, actions, prior_a, prior_W, prior_b)

    # Final marginals
    cache = CTCache(q_a, q_W, meta, q_b)
    alphas, betas, _, _ = forward_backward(prior_x, vae_msgs, cache, actions)
    marginals = compute_marginals(alphas, betas, vae_msgs)

    marginal_means, marginal_covs = [], []
    for q in marginals:
        mu, cov = gaussian_mean_cov(q)
        marginal_means.append(mu)
        marginal_covs.append(cov)

    return q_a, q_W, q_b, jnp.stack(marginal_means), jnp.stack(marginal_covs)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EMResult:
    """Holds Variational EM training outputs."""
    params: dict
    q_a: Gaussian
    q_W: Wishart
    q_b: Gaussian | None
    transition_matrix: jnp.ndarray
    transition_precision: jnp.ndarray
    control_matrix: jnp.ndarray | None
    loss_history: list[float]
    det_history: list[float]
    eigenvalues_history: list


# ---------------------------------------------------------------------------
# Main Variational EM
# ---------------------------------------------------------------------------

def variational_em(
    model,
    params: dict,
    trajectories: Sequence[dict],
    latent_dim: int,
    *,
    transform_fn: Callable | None = None,
    action_dim: int | None = None,
    n_em_iterations: int = 20,
    n_vmp_iterations: int = 20,
    n_m_steps: int = 10,
    lr: float = 1e-4,
    beta_recon: float = 1.0,
    prior_W_df: float = PRIOR_W_DF,
    prior_a_cov: float = PRIOR_A_COV,
    prior_a_mean: jnp.ndarray | None = None,
    init_a_cov: float = INIT_A_COV,
    prior_b_cov: float = PRIOR_B_COV,
    init_b_cov: float = INIT_B_COV,
    seed: int = 0,
    verbose: bool = True,
    callback: Callable | None = None,
) -> EMResult:
    """Variational EM: jointly learn VAE + dynamics from trajectory data.

    Parameters
    ----------
    model : VAE flax module
    params : initial VAE parameters
    trajectories : list of dicts, each with:
        - "observations": list of (28,28) images
        - "actions": list of (du,) arrays (optional, len = T-1)
    transform_fn : a → A reshape
    latent_dim : int
    action_dim : int, optional (required if trajectories have actions)
    """
    d = latent_dim
    da = d * d
    has_control = action_dim is not None
    rng = jax.random.PRNGKey(seed)
    if transform_fn is None:
        transform_fn = lambda a: a.reshape(d, d)
    meta = CTMeta(transform_fn)

    # Priors
    prior_a = gaussian_prior(da, prior_a_cov, prior_a_mean)
    q_a = gaussian_prior(da, init_a_cov, prior_a_mean)
    prior_W = Wishart(df=prior_W_df, inv_scale=jnp.eye(d))
    q_W = prior_W

    q_b, prior_b = None, None
    if has_control:
        db = d * action_dim
        prior_b = gaussian_prior(db, prior_b_cov)
        q_b = gaussian_prior(db, init_b_cov)

    # Collect all images for M-step
    all_images = []
    for traj in trajectories:
        all_images.extend(traj["observations"])
    images_batch = jnp.stack([jnp.array(img) for img in all_images])

    tx = optax.adam(lr)
    opt_state = tx.init(params)
    loss_history, det_history, eigenvalues_history = [], [], []

    @jax.jit
    def m_step_update(params, opt_state, images, post_means, post_covs, z_rng):
        loss, grads = jax.value_and_grad(_m_step_loss)(
            params, model, images, post_means, post_covs, z_rng, beta_recon,
        )
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    pbar = tqdm(range(1, n_em_iterations + 1), desc="EM", disable=not verbose)
    for em_it in pbar:
        # --- E-step: run inference on each trajectory ---
        encode_fn, _ = make_encode_decode(model, params)

        all_post_means, all_post_covs = [], []
        for traj in trajectories:
            obs = traj["observations"]
            acts = traj.get("actions", None)
            q_a, q_W, q_b, pmeans, pcovs = _e_step_trajectory(
                obs, acts, encode_fn, d,
                q_a, q_W, q_b, prior_a, prior_W, prior_b,
                meta, n_vmp_iterations,
            )
            all_post_means.append(pmeans)
            all_post_covs.append(pcovs)

        post_means = jnp.concatenate(all_post_means, axis=0)
        post_covs = jnp.concatenate(all_post_covs, axis=0)

        # --- M-step: update VAE ---
        for _ in range(n_m_steps):
            rng, z_rng = jax.random.split(rng)
            params, opt_state, loss = m_step_update(
                params, opt_state, images_batch, post_means, post_covs, z_rng,
            )
            loss_history.append(float(loss))

        ma = gaussian_mean(q_a)
        mA = transform_fn(ma)
        det_A = float(jnp.linalg.det(mA))
        det_history.append(det_A)
        eigenvalues_history.append(np.linalg.eigvals(np.array(mA)))

        postfix = {"loss": f"{float(loss):.1f}", "det(A)": f"{det_A:.4f}"}
        mB = None
        if has_control and q_b is not None:
            mb = gaussian_mean(q_b)
            mB = mb.reshape(d, action_dim)
            postfix["|B|"] = f"{float(jnp.linalg.norm(mb)):.3f}"
        pbar.set_postfix(postfix)

        if callback is not None:
            callback(em_it, params, mA, mB, float(loss), det_A)

    ma = gaussian_mean(q_a)
    mW = wishart_mean(q_W)
    control_matrix = None
    if has_control and q_b is not None:
        mb = gaussian_mean(q_b)
        control_matrix = mb.reshape(d, action_dim)

    return EMResult(
        params=params,
        q_a=q_a, q_W=q_W, q_b=q_b,
        transition_matrix=transform_fn(ma),
        transition_precision=mW,
        control_matrix=control_matrix,
        loss_history=loss_history,
        det_history=det_history,
        eigenvalues_history=eigenvalues_history,
    )
