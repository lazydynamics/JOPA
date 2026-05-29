"""Pendulum: image-only Bayesian inference and control — the full agent loop
on one factor graph.

  sense   — a (multi-frame) VAE encodes frames to a latent message
  learn   — `LearnedLinear` fits the latent dynamics x' = A·x + B·u + ε by
            Variational EM (E-step VMP on the chain, M-step refines the VAE)
  infer   — forward-backward smoothing of the latent state
  act     — `model.plan` infers the torque sequence (VMP on the actions) that
            drives the latent to a goal frame; receding-horizon execution

No hand-set physics, no linearisation tricks — just message passing. The
controller reaches the upright target frame; sustaining it there (an unstable
equilibrium) is beyond a single global linear-Gaussian model.

Example
-------
    uv run python examples/pendulum.py
    uv run python examples/pendulum.py --no-cache  # force retrain
"""
import argparse
import os
import pickle

import jax.numpy as jnp
import numpy as np

from jopa.envs import SimplePendulum
from jopa.nn.vae import VAE, train_vae, save_params, load_params
from jopa.blocks import JointModel, Block, LearnedLinear, LearnedVAE
from jopa.distributions import near_identity_prior, gaussian_mean

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--start-theta", type=float, default=0.0)
_p.add_argument("--goal-theta", type=float, default=float(np.pi))
_p.add_argument("--horizon", type=int, default=20,
                help="Planning horizon while far from the goal; shrinks to 2 near the goal.")
_p.add_argument("--exec-steps", type=int, default=2)
_p.add_argument("--n-replans", type=int, default=30)
_p.add_argument("--n-frames", type=int, default=4,
                help="Window size K for the multi-frame VAE encoder.")
_p.add_argument("--no-cache", action="store_true")
args = _p.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

K = args.n_frames
latent_dim = 4


# ──────────────────────────────────────────────────────────────────────────
# 1. Roll out training trajectories
# ──────────────────────────────────────────────────────────────────────────

env = SimplePendulum()
print("Generating training trajectories …")
n_trajectories, traj_len = 10, 80
train_frames, train_actions = [], []
for ep in range(n_trajectories):
    rng = np.random.RandomState(ep)
    env.reset(seed=ep)
    frames = [np.array(env.render())]
    actions = []
    for _ in range(traj_len - 1):
        torque = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]) + rng.randn() * 0.5
        actions.append(jnp.array([torque]))
        env.step(torque)
        frames.append(np.array(env.render()))
    train_frames.append(frames)
    train_actions.append(actions)
print(f"  {n_trajectories} trajectories × {traj_len} steps")


# ──────────────────────────────────────────────────────────────────────────
# 2. Pre-train multi-frame VAE
# ──────────────────────────────────────────────────────────────────────────

def windows_of(frames, K):
    """Sliding K-frame windows of a frame sequence."""
    arr = np.stack([np.asarray(f) for f in frames])
    return np.stack([arr[i:i + K] for i in range(len(arr) - K + 1)])


vae_path = os.path.join(CHECKPOINTS, f"vae_pendulum_d{latent_dim}_K{K}.npz")
vae_model = VAE(latent_dim=latent_dim, n_frames=K)
try:
    params = load_params(vae_model, vae_path)
    print(f"Loaded multi-frame VAE (K={K}) from {vae_path}")
except FileNotFoundError:
    print(f"Training multi-frame VAE (K={K}) …")
    windows = []
    for frames in train_frames:
        windows.append(windows_of(frames, K))     # (T-K+1, K, 28, 28)
    # Plus static "ω=0" coverage: K copies of the same frame across θ ∈ [-π, π].
    for theta in np.linspace(-np.pi, np.pi, 200):
        env.reset(theta=theta, theta_dot=0.0)
        f = np.array(env.render())
        windows.append(np.stack([f] * K)[None])
    windows = np.concatenate(windows, axis=0)
    # Reconstruct the WHOLE window (not just the last frame) → latent must encode motion.
    vae_model, params = train_vae(
        windows, latent_dim=latent_dim, n_frames=K, epochs=100, seed=42,
    )
    save_params(params, vae_path)
    print(f"Saved multi-frame VAE to {vae_path}")


# ──────────────────────────────────────────────────────────────────────────
# 3. Variational EM: learn latent dynamics
# ──────────────────────────────────────────────────────────────────────────

print("\n══ System Identification (Variational EM) ══")
em_cache = os.path.join(CHECKPOINTS, f"em_pendulum_K{K}.pkl")
priors = near_identity_prior(latent_dim, cov=0.5)
em_hparams = dict(
    n_em=30, n_vmp=10, n_m_steps=20, lr=5e-5, beta_recon=1.0,
    prior_b_cov=10.0, init_b_cov=100.0, seed=42, **priors,
)
fingerprint = {
    "vae_mtime": os.path.getmtime(vae_path),
    "K": K,
    "hparams": {k: (tuple(map(float, np.asarray(v).ravel())) if hasattr(v, "shape") else v)
                for k, v in em_hparams.items()},
}

cached = None
if not args.no_cache:
    try:
        with open(em_cache, "rb") as f:
            blob = pickle.load(f)
        if blob.get("fingerprint") == fingerprint:
            cached = blob
            print(f"Loaded EM result from {em_cache}")
        else:
            print(f"Cache fingerprint mismatch at {em_cache}; recomputing")
    except FileNotFoundError:
        pass

learned_vae = LearnedVAE(
    vae_model, params, lr=em_hparams["lr"],
    n_m_steps=em_hparams["n_m_steps"],
    beta_recon=em_hparams["beta_recon"], seed=em_hparams["seed"],
)
block = Block("z", LearnedLinear(
    dim=latent_dim, du=1, n_iterations=em_hparams["n_vmp"],
    mode="per_trajectory",
    prior_a_mean=priors["prior_a_mean"], prior_a_cov=priors["prior_a_cov"],
    init_a_cov=priors["init_a_cov"],
    prior_b_cov=em_hparams["prior_b_cov"], init_b_cov=em_hparams["init_b_cov"]),
    observe=learned_vae,
)
model = JointModel([block])

if cached is not None:
    block.transition.q_a = cached["q_a"]
    block.transition.q_W = cached["q_W"]
    block.transition.q_b = cached["q_b"]
    learned_vae.params = cached["vae_params"]
else:
    # Each trajectory's "z" observation sequence is its K-frame windows.
    trajs = []
    for frames, actions in zip(train_frames, train_actions):
        windows = list(windows_of(frames, K))
        # T-K+1 windows, T-K transitions → action sequence is actions[K-1:].
        trajs.append({"z": windows, "control": actions[K - 1:]})
    model.learn(trajs, n_em=em_hparams["n_em"])
    with open(em_cache, "wb") as f:
        pickle.dump({
            "fingerprint": fingerprint,
            "q_a": block.transition.q_a,
            "q_W": block.transition.q_W,
            "q_b": block.transition.q_b,
            "vae_params": learned_vae.params,
        }, f)
    print(f"Saved EM result to {em_cache}")

A = block.transition.A
B = block.transition.B
print(f"\n  det(A)={float(jnp.linalg.det(A)):.4f}")
print(f"  |B|={float(jnp.linalg.norm(B)):.4f}")


def save_control_gif(thetas, actions, frames, goal_theta, gif_path, png_path):
    """Animate the controlled pendulum toward the upright goal frame: rod+bob,
    the live observed frame, the |θ−goal| trace, and the applied torque
    sequence. Saves a GIF + a summary PNG (frozen at closest approach)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("(install matplotlib for the GIF:  uv pip install -e '.[viz]')")
        return
    thetas = np.asarray(thetas); actions = np.asarray(actions)
    e = np.abs((thetas - goal_theta + np.pi) % (2 * np.pi) - np.pi)
    n = len(thetas)
    gx, gy = float(np.sin(goal_theta)), float(-np.cos(goal_theta))

    fig = plt.figure(figsize=(8.5, 5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 1],
                          hspace=0.6, wspace=0.25)
    axp = fig.add_subplot(gs[:, 0]); axf = fig.add_subplot(gs[0, 1])
    axe = fig.add_subplot(gs[1, 1]); axu = fig.add_subplot(gs[2, 1])
    axp.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.4), aspect="equal")
    axp.set_title("image-goal control", fontsize=11)
    axp.set_xticks([]); axp.set_yticks([])
    axp.scatter([gx], [gy], marker="*", s=340, color="#d62728", zorder=5)  # goal (upright)
    axp.scatter([0], [0], s=30, color="0.4", zorder=4)
    trail, = axp.plot([], [], "-", color="#1f77b4", alpha=0.25, lw=2, zorder=2)
    rod, = axp.plot([], [], lw=3, color="#1f77b4", zorder=3)
    bob = axp.scatter([], [], s=170, color="#1f77b4", edgecolors="white", linewidths=1.3, zorder=4)
    axf.set_title("observed frame", fontsize=9); axf.set_xticks([]); axf.set_yticks([])
    im = axf.imshow(np.asarray(frames[0]), cmap="gray", vmin=0, vmax=1)
    axe.plot(e, color="#1f77b4", lw=1.5); axe.axhline(0.0, color="#d62728", ls="--", lw=1)
    axe.set(xlim=(0, max(n - 1, 1)), ylim=(-0.1, np.pi + 0.1))
    axe.set_ylabel("|θ−goal|", fontsize=8); axe.tick_params(labelsize=7, labelbottom=False)
    cur, = axe.plot([], [], "o", color="#1f77b4", ms=5)
    axu.step(np.arange(len(actions)), actions, where="post", color="#2ca02c", lw=1.3)
    axu.axhline(0.0, color="0.6", lw=0.8)
    axu.set(xlim=(0, max(n - 1, 1))); axu.set_ylabel("torque u", fontsize=8)
    axu.set_xlabel("control step", fontsize=8); axu.tick_params(labelsize=7)
    curu, = axu.plot([], [], "o", color="#2ca02c", ms=5)

    def upd(i):
        x, y = float(np.sin(thetas[i])), float(-np.cos(thetas[i]))
        rod.set_data([0, x], [0, y]); bob.set_offsets([[x, y]])
        lo = max(0, i - 20)
        trail.set_data(np.sin(thetas[lo:i + 1]), -np.cos(thetas[lo:i + 1]))
        im.set_data(np.asarray(frames[min(i, len(frames) - 1)]))
        cur.set_data([i], [e[i]])
        j = min(i, len(actions) - 1)
        curu.set_data([j], [actions[j]])
        return rod, bob, trail, im, cur, curu

    anim = FuncAnimation(fig, upd, frames=n, interval=90, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=12))
    upd(int(np.argmin(e)))                      # freeze on the closest-to-goal moment
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"  saved {gif_path}")
    print(f"  saved {png_path}  (closest err={e.min():.3f}; torque∈[{actions.min():+.1f}, {actions.max():+.1f}])")


# ──────────────────────────────────────────────────────────────────────────
# 4. Receding-horizon MPC
# ──────────────────────────────────────────────────────────────────────────

print("\n══ Planning as Inference (receding horizon) ══")

start_theta = args.start_theta
goal_theta = args.goal_theta
T_horizon = args.horizon
N_replan = args.n_replans
exec_steps = args.exec_steps


def err(theta):
    return abs((theta - goal_theta + np.pi) % (2 * np.pi) - np.pi)


# Goal "window": K copies of the upright frame (pendulum at goal, stationary).
env_goal = SimplePendulum(); env_goal.reset(theta=goal_theta, theta_dot=0.0)
goal_window = np.stack([np.array(env_goal.render())] * K)

env.reset(theta=start_theta, theta_dot=0.0)
# Initial buffer: K copies of the start frame (no motion observed yet).
buffer = [np.array(env.render())] * K
thetas_log = [float(env.state[0])]
actions_log = []

# Horizon by *proximity to the goal* (image-only): far away → plan a long swing
# to the goal; once the latent is near the goal latent → shrink to h=2 (tight
# "be there now" regulation).
mu_goal = gaussian_mean(learned_vae.message(goal_window))
for cycle in range(N_replan):
    current_window = np.stack(buffer[-K:])
    dist = float(jnp.linalg.norm(gaussian_mean(learned_vae.message(current_window)) - mu_goal))
    h = 2 if dist < 0.8 else T_horizon
    obs = {"z": [current_window] + [None] * (h - 2) + [goal_window]}
    actions = model.plan(obs, n_iterations=200)

    for i in range(min(exec_steps, actions.shape[0])):
        u = float(np.clip(actions[i][0], -env.max_torque, env.max_torque))
        env.step(u)
        buffer.append(np.array(env.render()))
        thetas_log.append(float(env.state[0]))
        actions_log.append(u)

    theta_now = env.state[0]
    print(f"  cycle {cycle + 1:2d}/{N_replan}: dist={dist:.2f} h={h:2d}  θ={theta_now:+.3f}  "
          f"err={err(theta_now):.3f}  u={actions_log[-1]:+.2f}")

print(f"\n  Final: θ={env.state[0]:.3f}  err={err(env.state[0]):.3f}")

save_control_gif(thetas_log, actions_log, buffer[K - 1:], goal_theta,
                 os.path.join(OUTPUTS, "pendulum.gif"),
                 os.path.join(OUTPUTS, "pendulum_control.png"))
