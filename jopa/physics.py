"""Grounded pendulum physics: learn it from proprioception, plan with it.

State s = (θ, ω). Known kinematics θ' = θ + Δt·ω'; force linear in known basis
features [sinθ, τ, ω]:

    ω' = ω + Δt·(k₁·sinθ + k₂·τ + k₃·ω)

`learn_physics` recovers k = (k₁,k₂,k₃) by conjugate Bayesian linear regression.
`linearize` produces a local linear-Gaussian transition (with the sinθ offset
carried in an augmented control input [τ, 1]) that the existing message-passing
planner can consume directly — enabling receding-horizon stabilization.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .envs import SimplePendulum
from .distributions import (
    Gaussian, Wishart, gaussian_prior, gaussian_mean, gaussian_mean_cov, wishart_mean,
)
from .nodes.transition import CTMeta, CTCache
from .message_passing import accumulate_vmp_messages

DT = 0.05  # SimplePendulum.dt


# ---------------------------------------------------------------------------
# A.1 — learn the physics from proprioceptive rollouts
# ---------------------------------------------------------------------------

def pendulum_rollout(n_traj=20, traj_len=200, seed=0, dt=DT):
    """Random-torque rollouts; returns (θ, ω, τ) arrays of shape (n_traj, T)."""
    thetas, omegas, taus = [], [], []
    for ep in range(n_traj):
        rng = np.random.RandomState(ep + seed)
        env = SimplePendulum(dt=dt)
        env.reset(seed=ep + seed)
        th, om, ta = [], [], []
        for _ in range(traj_len):
            theta, omega = env.state
            torque = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]) + rng.randn() * 0.5
            th.append(theta); om.append(omega); ta.append(torque)
            env.step(torque)
        thetas.append(th); omegas.append(om); taus.append(ta)
    return np.array(thetas), np.array(omegas), np.array(taus)


def learn_physics(thetas, omegas, taus, dt=DT, prior_std=10.0,
                  obs_prec=1e6, n_iter=5):
    """Learn k = (k₁,k₂,k₃) + process noise through the Continuous Transition
    node — the *same* VMP machinery JOPA uses everywhere else, not a bespoke
    regression.

    With proprioception the regressors x=[sinθ, ω] and target y=ω' are
    *observed*, so we feed the CT factor tight (delta-like) Gaussian messages on
    x and y and accumulate its conjugate :a / :b / :W messages:

        ω' = A·[sinθ, ω] + B·τ + ε,   A=(dt·k₁, 1+dt·k₃),  B=(dt·k₂),  ε~N(0,W⁻¹)

    Returns (mean k, std k, noise_std).
    """
    th = thetas[:, :-1].ravel()
    om = omegas[:, :-1].ravel()
    ta = taus[:, :-1].ravel()
    omn = omegas[:, 1:].ravel()
    keep = np.abs(omn) < 7.99                      # drop ω-clip transitions
    th, om, ta, omn = th[keep], om[keep], ta[keep], omn[keep]
    n = th.shape[0]

    feats = jnp.asarray(np.stack([np.sin(th), om], axis=1))   # (n, 2) = x side
    y = jnp.asarray(omn)[:, None]                             # (n, 1) = y side
    actions = jnp.asarray(ta)[:, None]                        # (n, 1) = control τ

    # Tight (observed) messages on x and y — stacked Gaussians.
    m_xs = Gaussian(eta=obs_prec * feats, lam=jnp.broadcast_to(obs_prec * jnp.eye(2), (n, 2, 2)))
    m_ys = Gaussian(eta=obs_prec * y, lam=jnp.broadcast_to(obs_prec * jnp.eye(1), (n, 1, 1)))

    meta = CTMeta(lambda a: a.reshape(1, 2))                  # A is (dy=1, dx=2)
    prior_a = gaussian_prior(2, prior_std ** 2)
    prior_b = gaussian_prior(1, prior_std ** 2)
    prior_W = Wishart(df=2.0, inv_scale=jnp.eye(1))
    q_a, q_W, q_b = prior_a, prior_W, prior_b
    for _ in range(n_iter):
        cache = CTCache(q_a, q_W, meta, q_b)
        q_a, q_W, q_b = accumulate_vmp_messages(
            m_xs, m_ys, cache, actions, prior_a, prior_W, prior_b)

    A = np.array(gaussian_mean(q_a))               # (2,)  [dt·k₁, 1+dt·k₃]
    B = float(gaussian_mean(q_b)[0])               # dt·k₂
    _, Va = gaussian_mean_cov(q_a)
    _, Vb = gaussian_mean_cov(q_b)
    sa = np.sqrt(np.diag(np.array(Va)))
    sb = float(np.sqrt(np.array(Vb)[0, 0]))

    k = np.array([A[0] / dt, B / dt, (A[1] - 1.0) / dt])
    k_std = np.array([sa[0] / dt, sb / dt, sa[1] / dt])
    noise_std = float(np.sqrt(1.0 / np.array(wishart_mean(q_W))[0, 0]))
    return k, k_std, noise_std


# ---------------------------------------------------------------------------
# A.2 — local linearization for the message-passing planner
# ---------------------------------------------------------------------------

def linearize(theta0, k, dt=DT):
    """Linearize the (semi-implicit Euler) pendulum at θ₀ into a 2-D
    linear-Gaussian transition with control u = [τ, 1]:

        s' = A s + B u,   s = (θ, ω),   u = (τ, 1)

    B's second column carries the constant offset from linearizing sinθ, so the
    affine term is handled without an augmented state dimension.
    """
    k1, k2, k3 = k
    g2 = dt * k1 * np.cos(theta0)                       # ∂(Δt·k₁ sinθ)/∂θ
    d = dt * k3
    gb = dt * k2
    c = dt * k1 * (np.sin(theta0) - np.cos(theta0) * theta0)   # offset
    A = np.array([
        [1.0 + dt * g2, dt * (1.0 + d)],
        [g2,            1.0 + d],
    ])
    B = np.array([
        [dt * gb, dt * c],
        [gb,      c],
    ])
    return A, B


def step_true(theta, omega, tau, k, dt=DT):
    """One step of the learned (nonlinear) physics — for sanity checks."""
    k1, k2, k3 = k
    omega_next = omega + dt * (k1 * np.sin(theta) + k2 * tau + k3 * omega)
    theta_next = theta + dt * omega_next
    return theta_next, omega_next
