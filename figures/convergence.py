"""Distance to goal over a closed-loop episode, and the agent's own spread.

    MUJOCO_GL=egl python figures/convergence.py --out assets/goal_convergence.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import BRICK, INK, INK_LIGHT, PAPER, RULE, SLATE, _bare_axes

from examples.reacher.runtime import ReacherLoop


def collect(seed, poses, steps):
    loop = ReacherLoop()
    runs = []
    for index in range(poses):
        run = loop.episode(seed + index, steps=steps,
                           track_uncertainty=True)
        agent = run["agent"]
        runs.append({
            "error": run["error"], "start_cm": run["start_cm"],
            "belief": float(np.sqrt(np.trace(
                np.asarray(agent.last_belief_cov)) / agent.transition.dim)),
        })
        print(f"pose {index}: start {run['start_cm']:5.1f} cm  "
              f"min {run['error'].min():5.2f}  "
              f"terminal {run['error'][-20:].mean():5.2f} cm", flush=True)
    loop.close()
    return runs


def collect_uncertainty(seed, steps):
    """One episode, keeping the belief and action spread at every step."""
    loop = ReacherLoop()
    start_pose, goal_pose = loop.sample_task(seed)
    goal_xy = loop.fingertip_of(goal_pose)
    loop.reset(goal_pose, goal_xy)
    goal_frame = loop.sensor_frame()
    agent = loop.agent_for(goal_frame, track_uncertainty=True)
    loop.reset(start_pose, goal_xy)
    history = [loop.sensor_frame()] * 4
    belief, action, error = [], [], []
    for _ in range(steps):
        control = agent.step(np.stack(history[-4:]))
        loop.act(control)
        history.append(loop.sensor_frame())
        belief.append(float(np.sqrt(np.trace(
            np.asarray(agent.last_belief_cov)) / agent.transition.dim)))
        action.append(float(np.sqrt(np.trace(
            np.asarray(agent.last_action_cov)) / agent.du)))
        error.append(float(np.linalg.norm(loop.fingertip() - goal_xy) * 100.0))
    loop.close()
    return np.array(belief), np.array(action), np.array(error)


def render(runs, belief, action, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10.4, 3.5), facecolor=PAPER)
    left = figure.add_axes([0.065, 0.19, 0.40, 0.68])
    right = figure.add_axes([0.585, 0.19, 0.37, 0.68])
    steps = len(runs[0]["error"])

    _bare_axes(left)
    closest = min(runs, key=lambda r: r["error"][-20:].mean())
    band_low = np.minimum.reduce([r["error"] for r in runs])
    band_high = np.maximum.reduce([r["error"] for r in runs])
    span = np.arange(steps)
    left.fill_between(span, band_low, band_high, color=SLATE, alpha=0.18,
                      linewidth=0)
    left.plot(span, closest["error"], color=INK, lw=1.6)
    left.axhline(3.0, color=BRICK, lw=1.1)
    left.set_xlim(0, steps - 1)
    left.set_ylim(0, max(r["start_cm"] for r in runs) * 1.12)
    left.set_xlabel("closed-loop step", color=INK_LIGHT, fontsize=9)
    left.set_ylabel("distance to goal (cm)", color=INK_LIGHT, fontsize=9)
    left.text(steps * 0.99, 3.9, "3 cm tolerance", color=BRICK, fontsize=8.5,
              ha="right", va="bottom")
    left.text(steps * 0.55, band_high[int(steps * 0.55)] + 1.5,
              f"range over {len(runs)} poses", color=INK_LIGHT, fontsize=8.5,
              ha="center", va="bottom")
    left.text(steps * 0.30, closest["error"][int(steps * 0.30)] + 4.0,
              "one pose", color=INK, fontsize=8.5, ha="center", va="bottom")

    _bare_axes(right)
    terminal = [r["error"][-20:].mean() for r in runs]
    order = np.argsort(terminal)
    right.axvline(3.0, color=BRICK, lw=1.1)
    for row, index in enumerate(order):
        right.plot([terminal[index]], [row], marker="o", ms=6, color=INK)
        right.text(terminal[index] + 0.12, row,
                   f"{runs[index]['start_cm']:.0f} cm start",
                   color=INK_LIGHT, fontsize=8.5, va="center")
    right.set_ylim(-0.7, len(runs) - 0.3)
    right.set_xlim(0, 4.2)
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    right.set_xlabel("terminal distance to goal (cm)", color=INK_LIGHT,
                     fontsize=9)
    right.text(3.05, len(runs) - 0.5, "tolerance", color=BRICK, fontsize=8.5,
               va="top")
    figure.savefig(path, dpi=200, facecolor=PAPER)
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="assets/goal_convergence.png")
    parser.add_argument("--seed", type=int, default=909090)
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args(argv)

    runs = collect(args.seed, args.poses, args.steps)
    belief, action, _ = collect_uncertainty(args.seed, args.steps)
    print(render(runs, belief, action, args.out), flush=True)


if __name__ == "__main__":
    main()
