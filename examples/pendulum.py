"""Pendulum: system identification + planning as inference.

Phase 1 — Pre-train VAE observation model from random trajectories.
Phase 2 — Variational EM: learn dynamics (A, B, W) via message passing.
Phase 3 — Receding-horizon MPC: plan actions via message passing to reach goal.

Example
-------
    uv run python examples/pendulum.py
    uv run python examples/pendulum.py --goal-theta 1.57 --n-replans 20
    uv run python examples/pendulum.py --no-cache  # force EM retrain
"""
import argparse
import os
import pickle
import jax.numpy as jnp
import numpy as np

from jopa.envs import SimplePendulum
from jopa.nn.vae import VAE, train_vae, save_params, load_params, make_encode_decode
from jopa.em import variational_em, Trajectory
from jopa.inference import plan
from jopa.distributions import near_identity_prior

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--start-theta", type=float, default=0.0,
                help="Initial pendulum angle in radians (default: 0.0 = hanging down).")
_p.add_argument("--goal-theta", type=float, default=float(np.pi),
                help="Target angle in radians (default: π = upright).")
_p.add_argument("--horizon", type=int, default=8,
                help="Planning horizon (steps per replan, default: 8).")
_p.add_argument("--exec-steps", type=int, default=2,
                help="Actions executed per replan cycle (default: 2).")
_p.add_argument("--n-replans", type=int, default=11,
                help="Number of observe-plan-act cycles (default: 11).")
_p.add_argument("--no-cache", action="store_true",
                help="Ignore any cached EM result and retrain from scratch.")
args = _p.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINTS, exist_ok=True)

latent_dim = 4
env = SimplePendulum()

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
        acts.append(jnp.array([torque]))
        env.step(torque)
        obs.append(jnp.array(env.render()))
    trajectories.append(Trajectory(observations=obs, actions=acts))

total_frames = sum(len(t.observations) for t in trajectories)
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
        all_frames.extend([np.array(o) for o in traj.observations])
    for theta in np.linspace(-np.pi, np.pi, 200):
        env.reset(theta=theta, theta_dot=0.0)
        all_frames.append(env.render())
    images = np.stack(all_frames)
    model, params = train_vae(images, latent_dim=latent_dim, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")

# ── 3. Variational EM: learn A, B, W via message passing ─────────────────
print("\n══ System Identification (Variational EM) ══")
em_cache = os.path.join(CHECKPOINTS, "em_result_pendulum.pkl")

em_hparams = dict(
    n_em_iterations=30, n_vmp_iterations=10, n_m_steps=20, lr=5e-5,
    beta_recon=1.0, prior_b_cov=10.0, init_b_cov=100.0, seed=42,
    **near_identity_prior(latent_dim, cov=0.5),
)
fingerprint = {
    "vae_mtime": os.path.getmtime(vae_path),
    "hparams": {k: v for k, v in em_hparams.items()
                if k not in ("prior_a_mean", "init_a_mean")},
}

result = None
if args.no_cache:
    print(f"--no-cache: skipping {em_cache} and retraining")
else:
    try:
        with open(em_cache, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            result = cached["result"]
            print(f"Loaded EM result from {em_cache}")
        else:
            print(f"Cache fingerprint mismatch at {em_cache}; recomputing")
    except FileNotFoundError:
        pass
    except (pickle.UnpicklingError, EOFError, KeyError) as e:
        print(f"Cache at {em_cache} unreadable ({e!r}); recomputing")

if result is None:
    result = variational_em(
        model=model, params=params, trajectories=trajectories,
        latent_dim=latent_dim, action_dim=1,
        **em_hparams,
    )
    with open(em_cache, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "result": result}, f)
    print(f"Saved EM result to {em_cache}")

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

vae = make_encode_decode(model, params)

start_theta = args.start_theta
goal_theta = args.goal_theta
T_horizon = args.horizon
N_replan = args.n_replans
exec_steps = args.exec_steps

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
        vae=vae,
        q_a=result.q_a, q_W=result.q_W, q_b=result.q_b,
        action_dim=1,
        n_iterations=200, verbose=False,
    )

    # ── Act ──
    for i in range(min(exec_steps, len(plan_result.actions))):
        torque = float(plan_result.actions[i][0])
        env.step(torque)

    theta_now = env.state[0]
    error = abs((theta_now - goal_theta + np.pi) % (2 * np.pi) - np.pi)
    print(f"  cycle {cycle+1:2d}/{N_replan}: "
          f"θ={theta_now:+.3f}  error={error:.3f}")

_final_err = abs((env.state[0] - goal_theta + np.pi) % (2 * np.pi) - np.pi)
print(f"\n  Final: θ={env.state[0]:.3f} (goal={goal_theta:.3f}, error={_final_err:.3f})")
