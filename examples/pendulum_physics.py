"""Layer 3 / A.1 — learn pendulum physics from proprioception (conjugate VB).

Grounded state s = (θ, ω). Known kinematics  θ' = θ + Δt·ω.  Force is linear in
known basis features:

    ω' = ω + Δt·(k₁·sinθ + k₂·τ + k₃·ω)

so the force model is Bayesian *linear regression* in the params k = (k₁,k₂,k₃)
— conjugate, identifiable, and structurally able to represent the unstable top.
Ground truth for SimplePendulum (g=9.81, m=l=1, no damping): k₁=-9.81, k₂=1, k₃=0.
"""
import numpy as np

from jopa.envs import SimplePendulum

DT = 0.05  # SimplePendulum.dt


def rollout(n_traj=20, traj_len=200, seed=0):
    """Random-torque rollouts. Returns proprio arrays (θ, ω) and torques."""
    thetas, omegas, taus = [], [], []
    for ep in range(n_traj):
        rng = np.random.RandomState(ep + seed)
        env = SimplePendulum(dt=DT)
        env.reset(seed=ep + seed)
        th, om, ta = [], [], []
        for _ in range(traj_len):
            theta, omega = env.state
            torque = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]) + rng.randn() * 0.5
            th.append(theta); om.append(omega); ta.append(torque)
            env.step(torque)
        thetas.append(th); omegas.append(om); taus.append(ta)
    return np.array(thetas), np.array(omegas), np.array(taus)


def learn_physics(thetas, omegas, taus, prior_std=10.0):
    """Conjugate Bayesian linear regression for k = (k₁,k₂,k₃) and noise.

    Target  Δω = ω_{t+1} − ω_t = Δt·(k₁ sinθ + k₂ τ + k₃ ω) + ε.
    """
    th = thetas[:, :-1].ravel()
    om = omegas[:, :-1].ravel()
    ta = taus[:, :-1].ravel()
    dom = (omegas[:, 1:] - omegas[:, :-1]).ravel()

    # Drop transitions where the env's ω-clip (|ω|<8) was active — outside the model.
    keep = np.abs(omegas[:, 1:].ravel()) < 7.99
    th, om, ta, dom = th[keep], om[keep], ta[keep], dom[keep]

    X = DT * np.stack([np.sin(th), ta, om], axis=1)        # (N, 3)
    y = dom                                                # (N,)

    # Posterior: weak Gaussian prior N(0, prior_std²·I), noise precision β (MLE).
    alpha = 1.0 / prior_std ** 2
    # Iterate noise estimate once for a sane β.
    beta = 1.0
    for _ in range(3):
        S = np.linalg.inv(alpha * np.eye(3) + beta * X.T @ X)
        m = beta * S @ X.T @ y
        resid = y - X @ m
        beta = 1.0 / (resid.var() + 1e-12)
    std = np.sqrt(np.diag(S))
    return m, std, np.sqrt(1.0 / beta)


if __name__ == "__main__":
    print("Rolling out random-torque trajectories …")
    thetas, omegas, taus = rollout(n_traj=20, traj_len=200, seed=0)
    n_trans = thetas.shape[0] * (thetas.shape[1] - 1)
    print(f"  {thetas.shape[0]} trajectories × {thetas.shape[1]} steps = {n_trans} transitions")

    m, std, noise_std = learn_physics(thetas, omegas, taus)
    names = ["k1 (gravity, sinθ)", "k2 (control, τ)", "k3 (damping, ω)"]
    truth = [-9.81, 1.0, 0.0]
    print("\nLearned force coefficients (posterior mean ± std):")
    for nm, mi, si, tr in zip(names, m, std, truth):
        print(f"  {nm:22s}: {mi:+8.4f} ± {si:.4f}   (true {tr:+.2f})")
    print(f"  process-noise std on Δω: {noise_std:.4f}")
