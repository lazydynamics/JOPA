"""Collect additional motor-babbling replay for the pixel Reacher sensor.

Produces a pickle of (frames, controls) chunks in the same format, camera,
and control cadence as ``ck_babble.pkl`` / ``ck_replay.pkl`` (128x128
grayscale, target hidden, action repeat 2). Excitation policies are
state-free (held random torques, Ornstein-Uhlenbeck noise, and random
sinusoid mixtures); simulator coordinates are used only to place the initial
pose, exactly as the original babbling episode did. No simulator state is
stored.

Usage:
    MUJOCO_GL=egl python examples/reacher_collect_babble.py \
        --out outputs/gpu_run/ck_babble2.pkl --trajectories 900
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/gpu_run/ck_babble2.pkl")
    parser.add_argument("--trajectories", type=int, default=900)
    parser.add_argument("--steps", type=int, default=110)
    parser.add_argument("--img-size", type=int, default=128)
    # Seeds live far away from the preserved/protocol seeds (73001/73002/73108).
    parser.add_argument("--seed0", type=int, default=510000)
    parser.add_argument(
        "--profile", choices=("wild", "rest"), default="wild",
        help="wild: full-torque excitation; rest: bursts, coasts, and "
             "micro-torques — the low-speed regime precise holding uses")
    return parser


def hold_policy(rng, steps, hold=3):
    controls = np.empty((steps, 2), dtype=np.float32)
    u = rng.uniform(-1, 1, 2)
    for t in range(steps):
        if t % hold == 0:
            u = rng.uniform(-1, 1, 2)
        controls[t] = u
    return controls


def ou_policy(rng, steps, decay=0.85, drive=0.5):
    controls = np.empty((steps, 2), dtype=np.float32)
    u = rng.uniform(-1, 1, 2)
    for t in range(steps):
        u = np.clip(decay * u + drive * rng.randn(2), -1.0, 1.0)
        controls[t] = u
    return controls


def sine_policy(rng, steps):
    t = np.arange(steps)[:, None]
    omega = rng.uniform(0.05, 0.9, (2, 2))
    phase = rng.uniform(0, 2 * np.pi, (2, 2))
    amp = rng.uniform(0.3, 1.0, (2, 2))
    wave = sum(amp[k] * np.sin(omega[k] * t + phase[k]) for k in range(2))
    return np.clip(wave, -1.0, 1.0).astype(np.float32)


def burst_coast_policy(rng, steps):
    controls = np.zeros((steps, 2), dtype=np.float32)
    t = 0
    while t < steps:
        burst = rng.randint(2, 6)
        coast = rng.randint(8, 21)
        controls[t:t + burst] = rng.uniform(-1, 1, 2)
        t += burst + coast
    return controls


def micro_ou_policy(rng, steps, decay=0.95, drive=0.08):
    controls = np.empty((steps, 2), dtype=np.float32)
    u = rng.uniform(-0.2, 0.2, 2)
    for t in range(steps):
        u = np.clip(decay * u + drive * rng.randn(2), -0.4, 0.4)
        controls[t] = u
    return controls


def burst_then_micro_policy(rng, steps):
    controls = micro_ou_policy(rng, steps)
    burst = rng.randint(2, 5)
    controls[:burst] = rng.uniform(-1, 1, 2)
    return controls


POLICIES = (hold_policy, ou_policy, sine_policy)
REST_POLICIES = (burst_coast_policy, micro_ou_policy, burst_then_micro_policy)


def main(argv=None):
    args = build_parser().parse_args(argv)

    import gymnasium as gym
    import mujoco

    environment = gym.make("Reacher-v5", render_mode="rgb_array")
    world = environment.unwrapped.model
    data = environment.unwrapped.data
    target_id = world.geom("target").id
    renderer = mujoco.Renderer(world, args.img_size, args.img_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0]
    camera.distance, camera.elevation, camera.azimuth = 0.34, -90, 90

    def frame():
        world.geom_rgba[target_id, 3] = 0.0
        renderer.update_scene(data, camera=camera)
        return renderer.render().mean(2).astype(np.uint8)

    def act(u):
        data.ctrl[:] = np.clip(u, -1, 1)
        for _ in range(2):
            mujoco.mj_step(world, data)

    chunks = []
    try:
        for index in range(args.trajectories):
            rng = np.random.RandomState(args.seed0 + index)
            mujoco.mj_resetData(world, data)
            data.qpos[:2] = [rng.uniform(-np.pi, np.pi),
                             rng.uniform(-2.8, 2.8)]
            data.qpos[2:4] = [0.19, 0.19]
            rest = args.profile == "rest"
            data.qvel[:2] = (np.zeros(2) if rest or index % 2 == 0
                             else rng.uniform(-6.0, 6.0, 2))
            mujoco.mj_forward(world, data)
            policies = REST_POLICIES if rest else POLICIES
            controls = policies[index % len(policies)](rng, args.steps)
            frames = [frame()]
            for u in controls:
                act(u)
                frames.append(frame())
            chunks.append((np.stack(frames), controls))
            if (index + 1) % 100 == 0:
                print(f"collected {index + 1}/{args.trajectories}",
                      flush=True)
    finally:
        renderer.close()
        environment.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(chunks, handle)
    total = sum(len(c[0]) for c in chunks)
    print(f"saved {len(chunks)} trajectories ({total} frames) to {out}",
          flush=True)


if __name__ == "__main__":
    main()
