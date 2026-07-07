"""Boat on shifting currents: continual learning from pixels.

An image-only agent patrols waypoints on a small patch of water while the
physics change under it: a current switches on, then veers. The world
model — one factor graph — adapts by *inference, not fine-tuning*. Each part
of the posterior is treated according to what it is:

  structure    — q(A) is pinned by a kinematic prior ("positions persist"),
                 q(B) accumulates evidence forever: thrusters don't change
  calibration  — q(W), the noise model, is learned offline and held online
  disturbance  — q(c), the current, is the volatile bit: modeled as a slow
                 random walk, its covariance is diffused a little every
                 chunk, so fresh evidence can always re-aim it

Every 5-step chunk is absorbed by conjugate VMP (`learn`), folded into the
next prior (`remember`), and the drift block diffused — posterior in, prior
out, a little wider on the part of the world that can drift. `model.plan`
infers thrusts toward the goal frame (VMP on the actions), goal-pinned at
every step of the horizon, with a dash of exploration dither.

A frozen twin (no online updates) runs the same schedule: it station-keeps
confidently in calm water and gets blown off its goals when the current
arrives. No gradients on the dynamics, no replay buffer, no change-point
heuristics — the non-stationarity lives in the prior, where it belongs.

(An actuator-polarity flip is deliberately absent: recovering from it needs
uncertainty-driven exploration — expected free energy, issue #3.)

Example
-------
    uv run python examples/boat.py
    uv run python examples/boat.py --no-cache  # force retrain
"""
import argparse
import copy
import os
import pickle

import jax.numpy as jnp
import numpy as np

from jopa.envs import Boat
from jopa.nn.vae import VAE, train_vae, save_params, load_params
from jopa.blocks import JointModel, Block, LearnedLinear, LearnedVAE, Frozen, LinearCoupling
from jopa.distributions import near_identity_prior, gaussian_mean, gaussian_mean_cov

import boat_viz

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--horizon", type=int, default=8)
_p.add_argument("--exec-steps", type=int, default=2)
_p.add_argument("--chunk", type=int, default=5, help="Online update every CHUNK steps.")
_p.add_argument("--diffusion", type=float, default=2e-2,
                help="Per-chunk random-walk variance on the drift posterior q(c).")
_p.add_argument("--act-steps", type=int, default=100, help="Steps per regime act.")
_p.add_argument("--no-cache", action="store_true")
args = _p.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

latent_dim = 4
dt = 0.2

# The play in three acts: (name, current). The agent is never told.
ACTS = [("calm water", (0.0, 0.0)),
        ("current on", (0.4, 0.15)),
        ("current veers", (-0.15, 0.4))]
GOALS = [(-0.4, -0.4), (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (0.0, 0.0)]
GOAL_DWELL = 60
DITHER = 0.1                                   # exploration noise on actions


# ──────────────────────────────────────────────────────────────────────────
# 1. Roll out pretraining trajectories (calm water only)
# ──────────────────────────────────────────────────────────────────────────

def waypoint_rollout(env, T, rng):
    """Chase random waypoints — bounded, well-excited, exactly linear data."""
    frames, actions, states = [np.array(env.render())], [], [env.state.copy()]
    wp = rng.uniform(-0.7, 0.7, size=2)
    for t in range(T - 1):
        if t % 8 == 0:
            wp = rng.uniform(-0.7, 0.7, size=2)
        u = np.clip(2.0 * (wp - env.state) + 0.3 * rng.randn(2), -1.0, 1.0)
        actions.append(jnp.asarray(u))
        env.step(u)
        frames.append(np.array(env.render()))
        states.append(env.state.copy())
    return frames, actions, states


env = Boat()
print("Generating training trajectories (calm water) …")
n_trajectories, traj_len = 20, 60
train_frames, train_actions, train_states = [], [], []
for ep in range(n_trajectories):
    rng = np.random.RandomState(ep)
    env.reset(seed=ep)
    f, a, s = waypoint_rollout(env, traj_len, rng)
    train_frames.append(f)
    train_actions.append(a)
    train_states.append(s)
print(f"  {n_trajectories} trajectories × {traj_len} steps")


# ──────────────────────────────────────────────────────────────────────────
# 2. Pre-train the VAE
# ──────────────────────────────────────────────────────────────────────────

vae_path = os.path.join(CHECKPOINTS, f"vae_boat_d{latent_dim}.npz")
vae_model = VAE(latent_dim=latent_dim, n_frames=1)
try:
    params = load_params(vae_model, vae_path)
    print(f"Loaded VAE from {vae_path}")
except FileNotFoundError:
    print("Training VAE …")
    imgs = [np.stack(f) for f in train_frames]
    for gx in np.linspace(-0.9, 0.9, 14):      # static coverage across the water;
        for gy in np.linspace(-0.9, 0.9, 14):  # corners/edges oversampled — the
            env.reset(p=[gx, gy])              # KL otherwise shrinks rare extremes
            reps = 3 if max(abs(gx), abs(gy)) > 0.6 else 1
            imgs.append(np.repeat(np.array(env.render())[None], reps, axis=0))
    vae_model, params = train_vae(np.concatenate(imgs, axis=0), latent_dim=latent_dim,
                                  n_frames=1, epochs=100, seed=42)
    save_params(params, vae_path)
    print(f"Saved VAE to {vae_path}")


# ──────────────────────────────────────────────────────────────────────────
# 3. Variational EM: learn calm-water latent dynamics (with drift channel)
#
#    One strong structural prior: A stays near the identity — "positions
#    persist unless pushed" — so drift and thrust are credited to c and B,
#    not to a fictitious contraction of the latent space.
# ──────────────────────────────────────────────────────────────────────────

print("\n══ System Identification (Variational EM) ══")
em_cache = os.path.join(CHECKPOINTS, "em_boat.pkl")
priors = near_identity_prior(latent_dim, cov=1e-4)
em_hparams = dict(n_em=10, n_vmp=20, n_m_steps=20, lr=5e-5, beta_recon=1.0,
                  prior_b_cov=10.0, init_b_cov=100.0, seed=42, **priors)
fingerprint = {
    "vae_mtime": os.path.getmtime(vae_path),
    "static_anchors": True,
    "hparams": {k: (tuple(map(float, np.asarray(v).ravel())) if hasattr(v, "shape") else v)
                for k, v in em_hparams.items()},
}

cached = None
if not args.no_cache:
    try:
        with open(em_cache, "rb") as f:
            blob = pickle.load(f)
        if isinstance(blob, dict) and blob.get("fingerprint") == fingerprint:
            cached = blob
            print(f"Loaded EM result from {em_cache}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Ignoring unreadable EM cache at {em_cache} ({type(e).__name__}); recomputing")

learned_vae = LearnedVAE(vae_model, params, lr=em_hparams["lr"],
                         n_m_steps=em_hparams["n_m_steps"],
                         beta_recon=em_hparams["beta_recon"], seed=em_hparams["seed"])
transition = LearnedLinear(
    dim=latent_dim, du=2, offset=True, n_iterations=em_hparams["n_vmp"],
    prior_a_mean=priors["prior_a_mean"], prior_a_cov=priors["prior_a_cov"],
    init_a_cov=priors["init_a_cov"],
    prior_b_cov=em_hparams["prior_b_cov"], init_b_cov=em_hparams["init_b_cov"])
em_model = JointModel([Block("z", transition, observe=learned_vae)])

if cached is not None:
    transition.q_a, transition.q_W, transition.q_b = cached["q_a"], cached["q_W"], cached["q_b"]
    learned_vae.params = cached["vae_params"]
else:
    trajs = [{"z": list(f), "control": a} for f, a in zip(train_frames, train_actions)]
    # Static anchors: stationary pseudo-trajectories across the whole water —
    # they pin the encoder everywhere (banks included) through the M-step and
    # ground the drift channel at zero.
    for gx in np.linspace(-0.85, 0.85, 9):
        for gy in np.linspace(-0.85, 0.85, 9):
            env.reset(p=[gx, gy])
            f = np.array(env.render())
            trajs.append({"z": [f, f], "control": [jnp.zeros(2)]})
    em_model.learn(trajs, n_em=em_hparams["n_em"])
    with open(em_cache, "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "q_a": transition.q_a,
                     "q_W": transition.q_W, "q_b": transition.q_b,
                     "vae_params": learned_vae.params}, f)
    print(f"Saved EM result to {em_cache}")

print(f"  det(A)={float(jnp.linalg.det(transition.A)):.4f}  "
      f"|B|={float(jnp.linalg.norm(transition.B)):.4f}  "
      f"|c|={float(jnp.linalg.norm(transition.c)):.4f}")


# ──────────────────────────────────────────────────────────────────────────
# 4. Diagnostics: a latent→position probe (for the plots only — the agent
#    never sees true positions) and a calibrated innovation covariance
#    (turns one-step prediction errors into an honest surprise scale)
# ──────────────────────────────────────────────────────────────────────────

Z = np.stack([np.array(gaussian_mean(learned_vae.message(f)))
              for traj in train_frames[:8] for f in traj])
S = np.stack([s for traj in train_states[:8] for s in traj])
probe, diag = LinearCoupling.fit("z", "s", Z, S)
print(f"  probe latent→position: rmse={diag['residual_rmse']:.3f}")

print("Calibrating innovation covariance on held-out calm rollouts …")
innovations = []
for seed in (100, 101, 102):
    env.reset(seed=seed)
    f, a, _ = waypoint_rollout(env, 40, np.random.RandomState(seed))
    msgs = [learned_vae.message(fr) for fr in f]
    for t in range(len(a)):
        mu_p, _ = gaussian_mean_cov(transition.predict(msgs[t], a[t]))
        innovations.append(np.array(gaussian_mean(msgs[t + 1])) - np.array(mu_p))
S_CAL_INV = np.linalg.inv(np.cov(np.stack(innovations).T) + 1e-6 * np.eye(latent_dim))


def current_posterior(tr):
    """q(c) pushed through the probe → current estimate (mean, cov)."""
    mb, Vb = gaussian_mean_cov(tr.q_b)
    eff = tr.eff_du
    c_z = mb.reshape(latent_dim, eff)[:, -1]
    idx = np.arange(latent_dim) * eff + (eff - 1)
    Vc = np.array(Vb)[np.ix_(idx, idx)]
    Mp = np.array(probe.M)
    return Mp @ np.array(c_z) / dt, Mp @ Vc @ Mp.T / dt ** 2


def innovation(tr, msg_prev, u_prev, msg_cur):
    """One-step prediction error of the encoded latent."""
    mu_p, _ = gaussian_mean_cov(tr.predict(msg_prev, u_prev))
    return np.array(gaussian_mean(msg_cur)) - np.array(mu_p)


def project(z_mean):
    """Latent mean → arena position, through the diagnostic probe."""
    return np.array(probe.M) @ np.array(z_mean) + np.array(probe.offset)


# ──────────────────────────────────────────────────────────────────────────
# 5. The stream: patrol goals while the water changes — adaptive vs frozen
# ──────────────────────────────────────────────────────────────────────────

def goal_img(p):
    g = Boat()
    g.reset(p=p)
    return np.array(g.render())


GOAL_IMGS = {g: goal_img(list(g)) for g in GOALS}


def run_stream(adapt):
    """One pass through the three acts. adapt=False freezes the world model."""
    tr = copy.deepcopy(transition)
    W_cal = tr.q_W                             # offline noise calibration
    model = JointModel([Block("z", tr,
                              observe=Frozen(learned_vae.message,
                                             decode=learned_vae.decode))])
    env = Boat()
    env.reset(p=[0.0, 0.0])
    rng = np.random.RandomState(7)
    n_steps = args.act_steps * len(ACTS)
    log = {k: [] for k in ("p", "goal", "u", "nis", "cur_mu", "cur_cov",
                           "act", "frame", "imag", "plan_path")}
    frame = np.array(env.render())
    msg_prev = u_prev = None
    chunk_msgs, chunk_us, chunk_shore = [], [], []
    actions = np.zeros((1, 2))

    for t in range(n_steps):
        act_i = t // args.act_steps
        env.current = np.asarray(ACTS[act_i][1], dtype=float)
        goal = GOALS[min(t // GOAL_DWELL, len(GOALS) - 1)]

        msg = learned_vae.message(frame)
        if msg_prev is not None:
            e = innovation(tr, msg_prev, u_prev, msg)
            log["nis"].append(float(e @ S_CAL_INV @ e))
        else:
            log["nis"].append(np.nan)
        chunk_msgs.append(msg)
        believed = project(gaussian_mean(msg))
        chunk_shore.append(bool(np.max(np.abs(believed)) > 0.65))

        # Online conjugate update: absorb the chunk, fold the posterior into
        # the next prior (noise calibration held — it is a sensor property),
        # and take one random-walk step on the drift. Chunks that touch the
        # shore are skipped — bank contact violates the open-water model.
        if adapt and len(chunk_msgs) == args.chunk + 1:
            if not any(chunk_shore):
                tr.learn([chunk_msgs], [chunk_us])
                tr.q_W = W_cal
                tr.remember(forget=1.0, diffuse=args.diffusion)
            chunk_msgs, chunk_us, chunk_shore = [msg], [], [chunk_shore[-1]]

        # Receding-horizon planning: the goal frame is pinned at every step
        # of the horizon — "be there as soon as possible, then stay".
        if t % args.exec_steps == 0:
            gimg = GOAL_IMGS[goal]
            actions = np.array(model.plan({"z": [frame] + [gimg] * (args.horizon - 1)},
                                          n_iterations=100, prior_x=msg))
            # Imagination: roll the belief through the plan and decode the
            # endpoint — prediction lives in latent space, pixels are a readout.
            bel, path = msg, [believed]
            for uu in actions:
                bel = tr.predict(bel, uu)
                path.append(project(gaussian_mean(bel)))
            imagined = np.array(learned_vae.decode(gaussian_mean(bel)))
            plan_path = np.asarray(path)

        u = np.clip(actions[min(t % args.exec_steps, len(actions) - 1)]
                    + DITHER * rng.randn(2), -env.max_thrust, env.max_thrust)
        env.step(u)
        frame = np.array(env.render())

        c_mu, c_cov = current_posterior(tr)
        log["imag"].append(imagined)
        log["plan_path"].append(plan_path)
        log["p"].append(env.state.copy())
        log["goal"].append(np.asarray(goal))
        log["u"].append(np.asarray(u))
        log["act"].append(act_i)
        log["cur_mu"].append(c_mu)
        log["cur_cov"].append(c_cov)
        log["frame"].append(frame)
        msg_prev, u_prev = msg, u
        chunk_us.append(jnp.asarray(u))

    return {k: (np.asarray(v) if k != "frame" else v) for k, v in log.items()}


print("\n══ The stream: three acts, two world models ══")
print("Running the adaptive agent (learn + remember every chunk) …")
log_on = run_stream(adapt=True)
print("Running the frozen twin (no online updates) …")
log_off = run_stream(adapt=False)

err_on = np.linalg.norm(log_on["p"] - log_on["goal"], axis=1)
err_off = np.linalg.norm(log_off["p"] - log_off["goal"], axis=1)
print(f"\n  {'':<18} {'——— full act ———':>20} {'—— settled half ——':>21}")
print(f"  {'act':<18} {'adaptive':>9} {'frozen':>9}  {'adaptive':>9} {'frozen':>9}")
for i, (name, cur) in enumerate(ACTS):
    m = log_on["act"] == i
    s = m.copy()
    s[:np.argmax(m) + args.act_steps // 2] = False    # second half of the act
    print(f"  {name:<18} {err_on[m].mean():>9.3f} {err_off[m].mean():>9.3f}"
          f"  {err_on[s].mean():>9.3f} {err_off[s].mean():>9.3f}")


# ──────────────────────────────────────────────────────────────────────────
# 6. Artifacts: money plot, hero animation, and the plan → act → adapt loop
# ──────────────────────────────────────────────────────────────────────────

if boat_viz.save_money_plot(os.path.join(OUTPUTS, "boat_adapt.png"),
                            log_on, log_off, err_on, err_off, ACTS, args.act_steps):
    boat_viz.save_hero_gif(os.path.join(OUTPUTS, "boat.gif"),
                           log_on, log_off, err_on, err_off, ACTS, args.act_steps)
    boat_viz.save_loop_gif(os.path.join(OUTPUTS, "boat_loop.gif"),
                           log_on, log_off, ACTS, args.act_steps,
                           args.chunk, args.exec_steps)
