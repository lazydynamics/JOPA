"""Layer 1 — online recursive filtering for receding-horizon control.

Compares two control loops on the same EM-learned model:

  baseline  — re-encode a single current frame each cycle (prior x₀ = N(0,I)),
              the original `examples/pendulum.py` behaviour.
  filtered  — carry a belief over the current state across cycles via a
              predict-update Bayes filter (push through the dynamics with the
              applied action, then fuse the new frame). The belief accumulates
              velocity-like information from the frame stream, so the planner
              knows its current velocity.

Goal is still given as an image here (Layer 2 will move to a setpoint).
"""
import os
import pickle

import jax.numpy as jnp
import numpy as np

from jopa.envs import SimplePendulum
from jopa.nn.vae import VAE, load_params, make_encode_decode
from jopa.inference import plan
from jopa.distributions import Gaussian, combine_gaussians
from jopa.nodes.transition import CTMeta, CTCache, ct_forward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

latent_dim = 4
goal_theta = np.pi
T_horizon = 8
N_replan = 30          # long enough to observe holding past first arrival
exec_steps = 2

model = VAE(latent_dim=latent_dim)
params = load_params(model, os.path.join(CHECKPOINTS, "vae_pendulum_d4.npz"))
vae = make_encode_decode(model, params)

with open(os.path.join(CHECKPOINTS, "em_result_pendulum.pkl"), "rb") as f:
    result = pickle.load(f)["result"]
q_a, q_W, q_b = result.q_a, result.q_W, result.q_b

meta = CTMeta(lambda a: a.reshape(latent_dim, latent_dim))
cache = CTCache(q_a, q_W, meta, q_b)

env_goal = SimplePendulum()
env_goal.reset(theta=goal_theta, theta_dot=0.0)
goal_img = jnp.array(env_goal.render())


def encode_msg(frame):
    """Encode a frame into a Gaussian observation message (information form)."""
    mu, log_std = vae.encode(frame)
    var = jnp.exp(2.0 * log_std)
    lam = jnp.diag(1.0 / var)
    return Gaussian(eta=lam @ mu, lam=lam)


def ang_err(theta):
    return abs((theta - goal_theta + np.pi) % (2 * np.pi) - np.pi)


def run_baseline():
    env = SimplePendulum()
    env.reset(theta=0.0, theta_dot=0.0)
    thetas = [float(env.state[0])]
    for _ in range(N_replan):
        cur = jnp.array(env.render())
        obs = [cur] + [None] * (T_horizon - 2) + [goal_img]
        pr = plan(obs, vae, q_a, q_W, q_b, 1, n_iterations=200, verbose=False)
        for i in range(min(exec_steps, len(pr.actions))):
            env.step(float(pr.actions[i][0]))
            thetas.append(float(env.state[0]))
    return np.array(thetas)


def run_filtered():
    env = SimplePendulum()
    env.reset(theta=0.0, theta_dot=0.0)
    thetas = [float(env.state[0])]
    # Initial belief: vague prior fused with the first frame.
    belief = combine_gaussians(
        Gaussian(eta=jnp.zeros(latent_dim), lam=jnp.eye(latent_dim)),
        encode_msg(jnp.array(env.render())),
    )
    for _ in range(N_replan):
        obs = [None] * (T_horizon - 1) + [goal_img]
        pr = plan(obs, vae, q_a, q_W, q_b, 1, prior_x=belief,
                  n_iterations=200, verbose=False)
        for i in range(min(exec_steps, len(pr.actions))):
            u = pr.actions[i]
            belief = ct_forward(belief, cache, u)              # predict
            env.step(float(u[0]))
            belief = combine_gaussians(belief, encode_msg(jnp.array(env.render())))  # update
            thetas.append(float(env.state[0]))
    return np.array(thetas)


def run_filtered_hold(n_hold=3):
    """Filtered belief + goal held over the final n_hold timesteps → the plan
    must arrive *and stay* (zero velocity), without any sensor."""
    env = SimplePendulum()
    env.reset(theta=0.0, theta_dot=0.0)
    thetas = [float(env.state[0])]
    belief = combine_gaussians(
        Gaussian(eta=jnp.zeros(latent_dim), lam=jnp.eye(latent_dim)),
        encode_msg(jnp.array(env.render())),
    )
    obs = [None] * (T_horizon - n_hold) + [goal_img] * n_hold
    for _ in range(N_replan):
        pr = plan(obs, vae, q_a, q_W, q_b, 1, prior_x=belief,
                  n_iterations=200, verbose=False)
        for i in range(min(exec_steps, len(pr.actions))):
            u = pr.actions[i]
            belief = ct_forward(belief, cache, u)
            env.step(float(u[0]))
            belief = combine_gaussians(belief, encode_msg(jnp.array(env.render())))
            thetas.append(float(env.state[0]))
    return np.array(thetas)


def summarize(name, thetas):
    errs = np.array([ang_err(t) for t in thetas])
    reached = errs.min()
    # First step that gets within 0.3 rad of the goal; mean error afterwards.
    within = np.where(errs < 0.3)[0]
    if len(within):
        arrival = int(within[0])
        post = errs[arrival:]
        hold = post.mean()
        hold_str = f"arrived@step {arrival}; post-arrival mean error {hold:.3f} (max {post.max():.3f})"
    else:
        hold_str = "never reached 0.3 rad"
    print(f"\n{name}:")
    print(f"  error trajectory: {np.array2string(errs, precision=2, max_line_width=130)}")
    print(f"  closest approach: {reached:.3f} rad   |   {hold_str}")


if __name__ == "__main__":
    print("Running baseline (re-encode each cycle) …")
    base = run_baseline()
    print("Running filtered (carried predict-update belief) …")
    filt = run_filtered()
    print("Running filtered + held goal (arrive and stay) …")
    hold = run_filtered_hold(n_hold=3)
    summarize("BASELINE", base)
    summarize("FILTERED", filt)
    summarize("FILTERED + HELD GOAL", hold)
