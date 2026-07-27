"""Pixel-only structured-state MuJoCo Reacher.

    python -m examples.reacher.run train    --data outputs/gpu_run
    python -m examples.reacher.run validate --data outputs/gpu_run
    MUJOCO_GL=egl python -m examples.reacher.run evaluate --pose-seed 515151

Training and diagnostics use agent-generated grayscale pixels and actions only.
MuJoCo coordinates enter in evaluate alone, to build reachable tasks and score
the final centimetre error.
"""
from __future__ import annotations

import argparse

import jax

from .diagnose import run_validate
from .evaluate import DEFAULT_POSE_SEED, SURVEY_POSES, run_evaluate
from .runtime import MOTION_DIM, N_FRAMES, POSE_DIM, SENSOR_CHANNELS
from .train import run_train


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("train", "validate", "evaluate"))
    parser.add_argument("--data", default="outputs/gpu_run")
    parser.add_argument("--outdir", default="outputs/reacher_pixels")
    parser.add_argument("--channels", type=int, default=SENSOR_CHANNELS)
    parser.add_argument("--rollout", type=int, default=12)
    parser.add_argument("--pose-steps", type=int, default=1500)
    parser.add_argument("--controlled-steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--pose-seed", type=int, default=DEFAULT_POSE_SEED)
    parser.add_argument("--poses", type=int, default=SURVEY_POSES)
    return parser



def main(argv=None):
    args = build_parser().parse_args(argv)
    print(
        f"jax backend: {jax.default_backend()}  "
        f"devices: {jax.devices()}", flush=True)
    if N_FRAMES != 4:
        raise ValueError(
            "the canonical structured sensor requires exactly four frames")
    if POSE_DIM != 4 or MOTION_DIM != 2:
        raise ValueError(
            "the canonical state is exactly pose[4] + motion[2]")
    if args.mode == "train":
        run_train(args)
    elif args.mode == "validate":
        run_validate(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
