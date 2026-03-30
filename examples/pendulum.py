"""Pendulum: system identification + planning as inference.

Phase 1 — Pre-train VAE observation model from random trajectories.
Phase 2 — Variational EM: learn dynamics (A, B, W) via message passing.
Phase 3 — Receding-horizon MPC: plan actions via message passing to reach goal.
"""
import os
import jax.numpy as jnp
import numpy as np

from jopa.envs import SimplePendulum
from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
from jopa.em import variational_em
from jopa.inference import plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)

latent_dim = 4
ACTION_SCALE = 15.0
env = SimplePendulum()
transform_fn = lambda a: a.reshape(latent_dim, latent_dim)

# ── 1. Generate training trajectories ──────────────────────────────────────
print("Generating training trajectories …")
n_trajectories = 10
traj_len = 80

trajectories = []
for ep in range(n_trajectories):
    rng = np.random.RandomState(ep)
    env.reset(seed=ep)
    obs = [jnp.array(env.render())]
    acts = []
    for t in range(traj_len - 1):
        torque = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]) + rng.randn() * 0.5
        acts.append(jnp.array([torque / ACTION_SCALE]))
        env.step(torque)
        obs.append(jnp.array(env.render()))
    trajectories.append({"observations": obs, "actions": acts})

total_frames = sum(len(t["observations"]) for t in trajectories)
print(f"  {n_trajectories} trajectories × {traj_len} steps = {total_frames} frames")

# ── 2. Pre-train VAE (observation model) ──────────────────────────────────
vae_path = os.path.join(CHECKPOINTS, "vae_pendulum_d4.npz")
model = VAE(latent_dim=latent_dim)

try:
    params = load_params(model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Pre-training VAE …")
    all_frames = []
    for traj in trajectories:
        all_frames.extend([np.array(o) for o in traj["observations"]])
    for theta in np.linspace(-np.pi, np.pi, 200):
        env.reset(theta=theta, theta_dot=0.0)
        all_frames.append(env.render())
    images = np.stack(all_frames)
    model, params = train_vae(images, latent_dim=latent_dim, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

# ── 3. Variational EM: learn A, B, W via message passing ─────────────────
print("\n══ System Identification (Variational EM) ══")
vec_I = jnp.eye(latent_dim).ravel()

result = variational_em(
    model=model, params=params, trajectories=trajectories,
    transform_fn=transform_fn, latent_dim=latent_dim, action_dim=1,
    n_em_iterations=30, n_vmp_iterations=10, n_m_steps=20, lr=5e-5,
    beta_recon=1.0, prior_a_mean=vec_I, prior_a_cov=0.5, init_a_cov=0.5,
    prior_b_cov=10.0, init_b_cov=100.0, seed=42,
)

A = result.transition_matrix
B = result.control_matrix
det_A = float(jnp.linalg.det(A))
params = result.params

print(f"\n  det(A)={det_A:.4f}")
print(f"  A:\n{A}")
print(f"  B: {B.T[0]}")
print(f"  |B|={float(jnp.linalg.norm(B)):.3f}")

# ── 4. Receding-horizon MPC via message passing ──────────────────────────
#
#   for each cycle:
#       observe  → encode current image
#       plan     → run VMP+BP to infer action sequence toward goal
#       act      → execute first 2 actions in the real environment
#       repeat
#
print("\n══ Planning as Inference (receding horizon) ══")

encode_fn, decode_fn = make_encode_decode(model, params)

start_theta = 0.0
goal_theta = np.pi
T_horizon = 8       # plan this many steps ahead
N_replan = 11        # number of observe-plan-act cycles
exec_steps = 2       # execute this many actions per cycle

# Encode goal image once
env_goal = SimplePendulum()
env_goal.reset(theta=goal_theta, theta_dot=0.0)
goal_img = jnp.array(env_goal.render())

# Reset to start
env.reset(theta=start_theta, theta_dot=0.0)

for cycle in range(N_replan):
    # ── Observe ──
    current_img = jnp.array(env.render())

    # ── Plan (message passing: BP on states + VMP on actions) ──
    observations = [current_img] + [None] * (T_horizon - 2) + [goal_img]
    plan_result = plan(
        observations=observations,
        encode_fn=encode_fn, decode_fn=decode_fn,
        q_a=result.q_a, q_W=result.q_W, q_b=result.q_b,
        transform_fn=transform_fn,
        latent_dim=latent_dim, action_dim=1,
        n_iterations=200, prior_u_cov=10.0, verbose=False,
    )

    # ── Act ──
    for i in range(min(exec_steps, len(plan_result.actions))):
        torque = float(plan_result.actions[i][0]) * ACTION_SCALE
        env.step(torque)

    theta_now = env.state[0]
    error = abs((theta_now - goal_theta + np.pi) % (2 * np.pi) - np.pi)
    print(f"  cycle {cycle+1:2d}/{N_replan}: "
          f"θ={theta_now:+.3f}  error={error:.3f}")

_final_err = abs((env.state[0] - goal_theta + np.pi) % (2 * np.pi) - np.pi)
print(f"\n  Final: θ={env.state[0]:.3f} (goal={goal_theta:.3f}, error={_final_err:.3f})")
