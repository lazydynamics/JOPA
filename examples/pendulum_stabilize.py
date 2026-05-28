"""A.2 — stabilize the pendulum with learned physics + setpoint goal.

Receding-horizon planning-as-inference on the grounded state s=(θ,ω):
each control step we re-linearize the learned physics at the current angle,
build a CTCache from the linear-Gaussian transition, and run the existing
VMP-on-torques planner to a *setpoint* (θ=π, ω=0) — the ω=0 being the velocity
target the image-goal could never express. Apply the first torque, repeat.

Two tests:
  HOLD     — start near the top; can it stay (reject the fall)?
  SWING-UP — start at the bottom; can it get there and hold?
"""
import os

import numpy as np
import jax
import jax.numpy as jnp

from jopa.envs import SimplePendulum
from jopa.physics import pendulum_rollout, learn_physics, linearize, DT
from jopa.distributions import (
    Gaussian, Wishart, combine_gaussians, gaussian_mean, gaussian_prior, vague_gaussian,
)
from jopa.nodes.transition import CTMeta, CTCache, ct_marginal_yx, ct_message_u
from jopa.message_passing import forward_backward

D, DU = 2, 2
META = CTMeta(lambda a: a.reshape(D, D))
PROCESS_STD = 0.1
GOAL_PREC = 100.0
MAX_TORQUE = 50.0


def make_cache(A, B):
    q_a = gaussian_prior(D * D, 1e-8, jnp.asarray(A).ravel())     # tight → point est.
    q_b = gaussian_prior(D * DU, 1e-8, jnp.asarray(B).ravel())
    P = jnp.eye(D) / PROCESS_STD ** 2
    df = 100.0
    q_W = Wishart(df=df, inv_scale=df * jnp.linalg.inv(P))         # E[W] = P
    return CTCache(q_a, q_W, META, q_b)


def tight_gaussian(mean, prec):
    lam = jnp.eye(len(mean)) * prec
    return Gaussian(eta=lam @ jnp.asarray(mean), lam=lam)


def plan_grounded(prior_x, goal_msg, cache, T_horizon, prior_u, n_iter=100):
    n_ct = T_horizon - 1
    # Running setpoint: the goal is the target at *every* future step, not just
    # a terminal deadline — this makes it a regulator (hold), not a reach-by-end
    # plan (which lets the controller procrastinate and the pendulum fall).
    obs = [vague_gaussian(D)] + [goal_msg] * (T_horizon - 1)
    q_us = Gaussian(
        eta=jnp.broadcast_to(prior_u.eta, (n_ct, DU)),
        lam=jnp.broadcast_to(prior_u.lam, (n_ct, DU, DU)),
    )

    def upd(m_y, m_x, u):
        q_yx = ct_marginal_yx(m_y, m_x, cache, u)
        return combine_gaussians(prior_u, ct_message_u(q_yx, cache))
    vupd = jax.vmap(upd)

    for _ in range(n_iter):
        u_means = jax.vmap(gaussian_mean)(q_us)
        _, _, m_xs, m_ys = forward_backward(prior_x, obs, cache, u_means)
        q_us = vupd(m_ys, m_xs, u_means)
    return jax.vmap(gaussian_mean)(q_us)   # (n_ct, 2): columns [τ, const]


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def control(k, start_theta, n_cycles, T_horizon=15, n_iter=120):
    env = SimplePendulum(dt=DT)
    env.reset(theta=start_theta, theta_dot=0.0)
    # Action prior: permissive on torque, pin the constant input to 1.
    prior_u = Gaussian(
        eta=jnp.array([0.0, 1e8]),
        lam=jnp.diag(jnp.array([1e-2, 1e8])),
    )
    thetas = [float(env.state[0])]
    for _ in range(n_cycles):
        theta, omega = float(env.state[0]), float(env.state[1])
        goal_theta = theta + wrap(np.pi - theta)         # nearest π representative
        # Linearize at the *setpoint* (the top): gives the stabilizing feedback
        # structure for the unstable equilibrium.
        A, B = linearize(goal_theta, k)
        cache = make_cache(A, B)
        prior_x = tight_gaussian([theta, omega], 1e3)
        goal_msg = tight_gaussian([goal_theta, 0.0], GOAL_PREC)
        u = plan_grounded(prior_x, goal_msg, cache, T_horizon, prior_u, n_iter)
        tau = float(np.clip(u[0, 0], -MAX_TORQUE, MAX_TORQUE))
        env.step(tau)
        thetas.append(float(env.state[0]))
    return np.array(thetas)


def report(name, thetas):
    err = np.abs(wrap(thetas - np.pi))
    spun = (err > 3.0).any()       # did it ever fall all the way / spin through?
    print(f"\n{name}: start-err={err[0]:.3f}  closest={err.min():.3f}  "
          f"hold(last 30 mean)={err[-30:].mean():.3f}  spun_through={spun}")
    print("  error: " + np.array2string(err, precision=2, max_line_width=130))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")


def save_figures(runs):
    """runs: dict name -> theta trajectory. Saves a θ(t) plot and a GIF."""
    os.makedirs(OUTPUTS, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(install matplotlib for figures)")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for name, th in runs.items():
        ax.plot(np.abs(wrap(th - np.pi)), label=name, linewidth=2)
    ax.axhline(0.0, color="k", ls="--", alpha=0.4, label="upright (goal)")
    ax.axhline(np.pi, color="r", ls=":", alpha=0.3, label="hanging (fell/spun)")
    ax.set(xlabel="control step", ylabel="|θ − π|  (rad)",
           title="Stabilization: reach the top and hold (never spins through)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "pendulum_stabilize.png"), dpi=150)
    print(f"Saved {OUTPUTS}/pendulum_stabilize.png")

    # GIF of the REACH+HOLD run: render the real pendulum at each θ.
    try:
        from PIL import Image
        th = runs["REACH+HOLD"]
        env = SimplePendulum(dt=DT)
        frames = []
        for t in th[::1]:
            env.state = np.array([t, 0.0])
            arr = np.clip(env.render() * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(arr, "L").resize((168, 168), Image.NEAREST))
        frames[0].save(os.path.join(OUTPUTS, "pendulum_stabilize.gif"),
                       save_all=True, append_images=frames[1:], duration=60, loop=0)
        print(f"Saved {OUTPUTS}/pendulum_stabilize.gif")
    except ImportError:
        print("(install pillow for the GIF)")


if __name__ == "__main__":
    print("Learning physics …")
    th, om, ta = pendulum_rollout(n_traj=20, traj_len=200, seed=0)
    k, kstd, noise = learn_physics(th, om, ta)
    print(f"  k = {np.array2string(k, precision=3)}  (true [-9.81, 1, 0])")

    # Stabilization: started in the upper region, drive toward the top and HOLD
    # (bounded error, never spins through) — the behaviour the no-physics,
    # image-goal controller could never achieve (it spun: err oscillated 0↔3.14).
    print("\n── HOLD (start near top, θ=π−0.3) ──")
    hold = control(k, start_theta=np.pi - 0.3, n_cycles=100)
    report("HOLD-near", hold)

    print("\n── REACH + HOLD (start θ=π−1.0) ──")
    reach = control(k, start_theta=np.pi - 1.0, n_cycles=120)
    report("REACH+HOLD", reach)

    save_figures({"HOLD-near": hold, "REACH+HOLD": reach})
