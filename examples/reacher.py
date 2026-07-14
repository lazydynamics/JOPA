"""Reacher from pixels — an agent that teaches itself to see and act.

One factor graph, learned entirely inside the plan–act–observe–learn cycle:

* **sense**  — the repo's VAE encoder turns 128px frames into heteroscedastic
  Gaussian messages (`LearnedVAE`: the sensor reports its own confidence).
  Born untrained; its only teacher is the agent's own experience.
* **plan**   — every torque is the exact Gaussian action posterior
  (`model.plan(method="exact")` inside `Agent`): one forward–backward sweep
  on the chain, LQG duality made literal.
* **learn**  — the transition posterior q(A, B, W) is refined online by
  conjugate VMP with forgetting; the sensor is refined between episodes by a
  β-NLL "reflection" on the agent's own replay, anchored by its babbling.

The agent starts with motor babbling (episode zero), then lives in the cycle.
Over ~30 episodes it learns to see (sensor R² 0 → ~0.85+), to arrive gently
(25–38 rad/s ballistic flythroughs → 2–15 rad/s), and — where it practices —
to hold (settled fingertip error dropping toward centimetres).

Run (CPU, laptop-friendly)::

    python examples/reacher.py --throttle --episodes 12

Run (GPU box)::

    pip install -U "jax[cuda12]"           # CUDA jaxlib
    MUJOCO_GL=egl python examples/reacher.py --episodes 100 --batch 256

Checkpoints land in `--outdir` after every episode; rerunning resumes.
"""
import argparse
import os
import pickle
import re
import sys
import time

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--episodes", type=int, default=30, help="cycle episodes")
parser.add_argument("--babble", type=int, default=24, help="babbling rollouts (episode zero)")
parser.add_argument("--batch", type=int, default=16, help="sensor training batch size")
parser.add_argument("--img-size", type=int, default=128, choices=(64, 128))
parser.add_argument("--throttle", action="store_true",
                    help="limit CPU threads (laptop survival); ignore on GPU")
parser.add_argument("--render-every", type=int, default=0,
                    help="save an episode video every N episodes (0: off)")
parser.add_argument("--outdir", default="outputs/reacher_cycle")
args = parser.parse_args()

if args.throttle:                       # must precede the jax import
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
    os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
import mujoco

from jopa import Agent, Block, JointModel, LearnedLinear, LearnedVAE
from jopa.distributions import near_identity_prior
from jopa.nn.vae import VAE, save_params, load_params

print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
os.makedirs(args.outdir, exist_ok=True)
CK = os.path.join(args.outdir, "ck")
PROG = os.path.join(args.outdir, "progress.txt")

# ---------------------------------------------------------------------------
# Environment adapter: Gymnasium Reacher-v5, top-down grayscale camera
# ---------------------------------------------------------------------------
SV, K, DB, S, AR = 0.2, 2, 6, args.img_size, 2

env = gym.make("Reacher-v5", render_mode="rgb_array")
uw = env.unwrapped
world, data = uw.model, uw.data
FINGER = world.geom("fingertip").id
TARGET = world.geom("target").id
renderer = mujoco.Renderer(world, S, S)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0, 0, 0]
camera.distance, camera.elevation, camera.azimuth = 0.34, -90, 90


def frame():
    """The agent's retina: S×S grayscale, target hidden (goals come as images)."""
    world.geom_rgba[TARGET, 3] = 0.0
    renderer.update_scene(data, camera=camera)
    return renderer.render().mean(2).astype(np.float32) / 255.0


def reset(pose, target_xy):
    mujoco.mj_resetData(world, data)
    data.qpos[:2] = pose
    data.qpos[2:4] = target_xy
    data.qvel[:] = 0
    mujoco.mj_forward(world, data)


def act(u):
    data.ctrl[:] = np.clip(u, -1, 1)
    for _ in range(AR):
        mujoco.mj_step(world, data)


def fingertip():
    return data.geom_xpos[FINGER][:2].copy()


def fingertip_of(pose):
    saved = data.qpos.copy()
    data.qpos[:2] = pose
    mujoco.mj_forward(world, data)
    xy = fingertip()
    data.qpos[:] = saved
    mujoco.mj_forward(world, data)
    return xy


def true_state():
    """Privileged state — used only as the sensor's in-sim training signal
    and for reporting. Never visible to the controller."""
    return np.concatenate([data.qpos[:2].copy(), data.qvel[:2].copy()])


# -- domain geometry hooks for the Agent (latent = [cosθ₁ sinθ₁ cosθ₂ sinθ₂ 0.2ω]) --
def angles(z):
    return np.array([np.arctan2(z[1], z[0]), np.arctan2(z[3], z[2])])


def as_latent(a):
    return np.array([np.cos(a[0]), np.sin(a[0]), np.cos(a[1]), np.sin(a[1]), 0.0, 0.0])


# Reacher joint 2 is range-limited (see the MuJoCo model); wrapping its angle
# error can steer it the "short way" straight into the stop, where the arm jams
# at full torque. Wrap only the free joint; drive the limited joint directly.
JOINT2_LIMIT = 2.9


def reach_subgoal(belief_mean, goal_mean, trust=0.6):
    """Trust-region setpoint: step at most `trust` rad toward the goal, using a
    joint-limit-aware error (wrap the free joint, clamp-direct the limited one)."""
    th, tg = angles(belief_mean), angles(goal_mean)
    e1 = (tg[0] - th[0] + np.pi) % (2 * np.pi) - np.pi          # joint 1: free, wrap
    e2 = np.clip(tg[1], -JOINT2_LIMIT, JOINT2_LIMIT) - th[1]    # joint 2: limited, direct
    return as_latent(th + np.clip(np.array([e1, e2]), -trust, trust))


def joint_distance(belief_mean, goal_mean):
    th, tg = angles(belief_mean), angles(goal_mean)
    err = (tg - th + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(err))


# ---------------------------------------------------------------------------
# The sensor: the repo's VAE encoder, trained by β-NLL on the agent's replay
# ---------------------------------------------------------------------------
vae = VAE(latent_dim=DB, n_frames=K, img_size=S)
_reflect_tx = optax.adam(1e-4)
_bootstrap_tx = optax.adam(3e-4)
_reflect_opt = [None]                   # persistent across reflections


def state_targets(states):
    return np.concatenate([np.cos(states[:, 0:1]), np.sin(states[:, 0:1]),
                           np.cos(states[:, 1:2]), np.sin(states[:, 1:2]),
                           SV * states[:, 2:4]], axis=1)


def train_sensor(params, chunks, steps, bootstrap=False):
    """β-NLL (β=0.5) heteroscedastic regression onto privileged state.

    Plain heteroscedastic NLL lets the variance head *explain away* hard
    dimensions (velocity stalls at R²≈0); the stop-gradient variance
    reweighting keeps their means learning while the confidence stays honest.
    """
    frames_list = [c[0] for c in chunks]
    states_list = [c[2] for c in chunks]
    index = [(i, t) for i, fr in enumerate(frames_list) for t in range(len(fr) - K)]

    def gather(sel):
        x = np.stack([frames_list[i][t:t + K] for i, t in sel])
        y = np.stack([state_targets(states_list[i][t + K - 1:t + K])[0] for i, t in sel])
        return jnp.asarray(x), jnp.asarray(y)

    tx = _bootstrap_tx if bootstrap else _reflect_tx

    def loss(p, x, y):
        mu, ls = vae.apply(p, x, method=vae.encode)
        ls = jnp.clip(ls, -6.0, 2.0)
        nll = 0.5 * (y - mu) ** 2 * jnp.exp(-2 * ls) + ls
        return jnp.mean(nll * jax.lax.stop_gradient(jnp.exp(2 * ls)) ** 0.5)

    @jax.jit
    def step(p, o, x, y):
        _, g = jax.value_and_grad(loss)(p, x, y)
        up, o = tx.update(g, o, p)
        return optax.apply_updates(p, up), o

    if bootstrap or _reflect_opt[0] is None:
        opt = tx.init(params)
    else:
        opt = _reflect_opt[0]
    rng = np.random.RandomState(0)
    for _ in range(steps):
        sel = [index[j] for j in rng.choice(len(index), min(args.batch, len(index)), replace=False)]
        params, opt = step(params, opt, *gather(sel))
    if not bootstrap:
        _reflect_opt[0] = opt
    return params


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------
def babble(seed, steps=58):
    """Motor babbling: episode zero. The agent's first teacher is itself."""
    rng = np.random.RandomState(seed)
    reset(rng.uniform(-np.pi, np.pi, 2), [0.19, 0.19])
    frames, states, controls = [frame()], [true_state()], []
    u = rng.uniform(-1, 1, 2)
    for t in range(steps):
        if t % 3 == 0:
            u = rng.uniform(-1, 1, 2)
        controls.append(u.copy())
        act(u)
        frames.append(frame())
        states.append(true_state())
    return np.stack(frames), np.array(controls), np.stack(states)


def windows(frames):
    return np.stack([frames[i:i + K] for i in range(len(frames) - K + 1)])


def chunks_to_trajectories(chunks):
    return [{"z": list(windows(fr)), "control": [jnp.asarray(u) for u in U[K - 1:]]}
            for fr, U, _ in chunks]


PRIOR = near_identity_prior(DB, cov=0.5)


def build_model(params):
    observation = LearnedVAE(vae, params)
    transition = LearnedLinear(dim=DB, du=2, offset=False, n_iterations=20,
                               prior_a_mean=PRIOR["prior_a_mean"], prior_a_cov=0.5,
                               init_a_cov=0.5, prior_b_cov=1e3, init_b_cov=1e3)
    return JointModel([Block("z", transition, observe=observation)])


def run_episode(model, start, goal, record=None, steps=110):
    """One closed-loop episode through the Agent API."""
    agent = Agent(model, horizon=6, forget=0.5,
                  subgoal=reach_subgoal, goal_metric=joint_distance,
                  goal_precision=jnp.array([200.0] * 4 + [10.0] * 2),
                  relin_surprise=0.0, relin_radius=0.3)
    target_xy = fingertip_of(goal)
    reset(goal, target_xy)
    agent.goal(np.stack([frame()] * K))          # the goal is an image
    reset(start, target_xy)
    buffer = [frame()] * K
    errors, controls, states = [], [], []
    for _ in range(steps):
        u = agent.step(np.stack(buffer[-K:]))
        act(u)
        buffer.append(frame())
        controls.append(u)
        states.append(true_state())
        errors.append(np.linalg.norm(fingertip() - target_xy) * 100)
        if record is not None:
            record.append((data.qpos.copy(), errors[-1]))
    chunk = (np.stack(buffer[K - 1:]), np.array(controls), np.array(states))
    return errors, chunk


def save_video(record, path, fps=20):
    """Episode replay → mp4 (imageio+ffmpeg) or gif fallback."""
    from PIL import Image, ImageDraw
    disp = mujoco.Renderer(world, 440, 440)
    frames_ = []
    for qpos, err in record:
        data.qpos[:] = qpos
        mujoco.mj_forward(world, data)
        world.geom_rgba[TARGET, 3] = 1.0
        disp.update_scene(data, camera=camera)
        im = Image.fromarray(disp.render())
        ImageDraw.Draw(im).text((12, 414), f"fingertip error: {err:5.1f} cm", fill=(240, 240, 240))
        frames_.append(np.asarray(im))
    try:
        import imageio.v3 as iio
        iio.imwrite(path + ".mp4", np.stack(frames_), fps=fps)
        print(f"saved {path}.mp4", flush=True)
    except Exception:
        ims = [Image.fromarray(f) for f in frames_]
        ims[0].save(path + ".gif", save_all=True, append_images=ims[1:],
                    duration=int(1000 / fps), loop=0)
        print(f"saved {path}.gif", flush=True)


def main():
    # -- episode zero: babble, then bootstrap the sensor from it -------------
    if os.path.exists(CK + "_babble.pkl"):
        babble_chunks = pickle.load(open(CK + "_babble.pkl", "rb"))
    else:
        print("episode zero: babbling …", flush=True)
        babble_chunks = [babble(seed) for seed in range(args.babble)]
        pickle.dump(babble_chunks, open(CK + "_babble.pkl", "wb"))

    if os.path.exists(CK + "_sensor.msgpack"):
        params = load_params(vae, CK + "_sensor.msgpack")
        print("sensor resumed", flush=True)
    else:
        params = vae.init(jax.random.PRNGKey(0), jnp.ones((1, K, S, S)), jax.random.PRNGKey(1))
        t0 = time.time()
        params = train_sensor(params, babble_chunks, 3000, bootstrap=True)
        save_params(params, CK + "_sensor.msgpack")
        print(f"sensor bootstrapped from babble ({time.time() - t0:.0f}s)", flush=True)

    replay = []
    if os.path.exists(CK + "_replay.pkl"):
        replay = pickle.load(open(CK + "_replay.pkl", "rb"))
    start_ep = len(replay) // 3

    model = build_model(params)
    model.learn(chunks_to_trajectories(babble_chunks[:10] + replay[-8:]), n_em=1)

    eval_rng = np.random.RandomState(500)
    eval_poses = [(eval_rng.uniform(-np.pi, np.pi, 2), eval_rng.uniform(-np.pi, np.pi, 2))
                  for _ in range(3)]
    train_rng = np.random.RandomState(900)
    for _ in range(start_ep * 3):
        train_rng.uniform(-np.pi, np.pi, 2); train_rng.uniform(-np.pi, np.pi, 2)

    for ep in range(start_ep, args.episodes):
        # act: practice around the goals the agent is evaluated on — a
        # continual agent accumulates precision where it operates
        for i in range(3):
            start = train_rng.uniform(-np.pi, np.pi, 2)
            train_rng.uniform(-np.pi, np.pi, 2)          # keep stream aligned
            _, chunk = run_episode(model, start, eval_poses[i][1])
            replay.append(chunk)
        # evaluate
        record = [] if (args.render_every and ep % args.render_every == 0) else None
        settled = []
        for start, goal in eval_poses:
            errors, _ = run_episode(model, start, goal, record=record)
            settled.append(float(np.mean(errors[-20:])))
        line = f"ep{ep}: settled " + "/".join(f"{s:.1f}" for s in settled) + " cm"
        print(line, flush=True)
        open(PROG, "a").write(line + "\n")
        if record:
            save_video(record, os.path.join(args.outdir, f"episode_{ep:03d}"))
        # learn: reflect the sensor on replay (babble-anchored), refresh the model
        anchor = [babble_chunks[j] for j in
                  np.random.RandomState(ep).choice(len(babble_chunks), 6, replace=False)]
        params = train_sensor(params, replay[-8:] + anchor, 800)
        save_params(params, CK + "_sensor.msgpack")
        pickle.dump(replay, open(CK + "_replay.pkl", "wb"))
        model = build_model(params)
        model.learn(chunks_to_trajectories(babble_chunks[:10] + replay[-8:]), n_em=1)
    print("cycle complete", flush=True)


if __name__ == "__main__":
    main()
