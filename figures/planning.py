"""The closed loop, one step at a time: what the agent plans and what it does.

Each frame is one control step. The arm is on the left. On the right, the plan
that produced the torque about to be applied: six future actions inferred in a
single exact sweep, each with its posterior spread. The first is executed, the
rest are discarded and re-inferred next step.

    MUJOCO_GL=egl python figures/planning.py --out assets/planning.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import ACCENT, INK, INK_LIGHT, PAPER, RULE, SLATE, _bare_axes

from examples.reacher.runtime import ReacherLoop


def collect(seed, steps, stride):
    loop = ReacherLoop(record=True)
    start_pose, goal_pose = loop.sample_task(seed)
    goal_xy = loop.fingertip_of(goal_pose)
    loop.reset(goal_pose, goal_xy)
    goal_frame = loop.sensor_frame()
    agent = loop.agent_for(goal_frame, track_uncertainty=True)
    loop.reset(start_pose, goal_xy)
    history = [loop.sensor_frame()] * 4

    frames = []
    error = []
    for step in range(steps):
        action = agent.step(np.stack(history[-4:]))
        plan = agent.last_plan
        image = loop.display_frame()
        loop.act(action)
        history.append(loop.sensor_frame())
        error.append(float(np.linalg.norm(loop.fingertip() - goal_xy) * 100.0))
        if step % stride == 0:
            physical = agent.du
            mean = np.asarray(plan.mean)[:, :physical]
            spread = np.sqrt(np.diagonal(
                np.asarray(plan.marginal_covariance),
                axis1=1, axis2=2))[:, :physical]
            frames.append({
                "image": image, "plan": mean, "spread": spread,
                "error": list(error), "step": step,
            })
    start_cm = float(np.linalg.norm(loop.fingertip_of(start_pose) - goal_xy) * 100)
    loop.close()
    return frames, start_cm, len(error)


def render(frames, start_cm, steps, path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    horizon, channels = frames[0]["plan"].shape
    reach = max(np.abs(f["plan"]).max() + f["spread"].max() for f in frames)

    figure = plt.figure(figsize=(10.4, 4.6), facecolor=PAPER)
    ax_arm = figure.add_axes([0.015, 0.06, 0.30, 0.80])
    ax_error = figure.add_axes([0.40, 0.60, 0.55, 0.28])
    ax_plan = figure.add_axes([0.40, 0.12, 0.55, 0.33])
    ax_arm.axis("off")

    figure.text(0.015, 0.955, "one control step", color=INK, fontsize=13,
                va="top")
    figure.text(0.40, 0.955,
                "filter the frame · infer six actions in one sweep · "
                "apply the first · repeat",
                color=INK_LIGHT, fontsize=9.5, va="top")

    def update(index):
        frame = frames[index]
        ax_arm.clear()
        ax_arm.axis("off")
        ax_arm.imshow(frame["image"])

        _bare_axes(ax_error)
        trace = frame["error"]
        ax_error.plot(np.arange(len(trace)), trace, color=INK, lw=1.5)
        ax_error.axhline(3.0, color=RULE, lw=1.0)
        ax_error.set_xlim(0, steps - 1)
        ax_error.set_ylim(0, start_cm * 1.1)
        ax_error.set_ylabel("cm to goal", color=INK_LIGHT, fontsize=9)
        ax_error.text(steps - 1, 3.4, "3 cm", color=INK_LIGHT, fontsize=8.5,
                      ha="right", va="bottom")
        ax_error.text(len(trace) - 1, trace[-1] + start_cm * 0.06,
                      f"{trace[-1]:.2f} cm", color=INK, fontsize=9,
                      ha="right", va="bottom")

        _bare_axes(ax_plan)
        steps_ahead = np.arange(horizon)
        plan, spread = frame["plan"], frame["spread"]
        ax_plan.axhline(0.0, color=RULE, lw=1.0)
        offsets = np.linspace(-0.16, 0.16, channels)
        for channel in range(channels):
            for k in steps_ahead:
                colour = ACCENT if k == 0 else SLATE
                position = k + offsets[channel]
                ax_plan.vlines(position, plan[k, channel] - spread[k, channel],
                               plan[k, channel] + spread[k, channel],
                               color=colour, lw=1.3, alpha=0.5)
                ax_plan.plot([position], [plan[k, channel]],
                             marker="o" if channel == 0 else "s", ms=5.0,
                             color=colour)
        ax_plan.text(horizon - 1, reach * 1.02, "joint 1 ○   joint 2 □",
                     color=INK_LIGHT, fontsize=8.5, ha="right", va="top")
        ax_plan.set_xlim(-0.5, horizon - 0.5)
        ax_plan.set_ylim(-reach * 1.15, reach * 1.15)
        ax_plan.set_xticks(list(steps_ahead))
        ax_plan.set_xlabel("steps ahead", color=INK_LIGHT, fontsize=9)
        ax_plan.set_ylabel("planned torque", color=INK_LIGHT, fontsize=9)
        ax_plan.text(0, reach * 1.02, "applied now", color=ACCENT,
                     fontsize=8.5, ha="center", va="top")
        ax_plan.text(horizon - 1, -reach * 1.02,
                     "inferred, then discarded", color=INK_LIGHT,
                     fontsize=8.5, ha="right", va="bottom")
        return []

    animation = FuncAnimation(figure, update, frames=len(frames), blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=110,
                   savefig_kwargs={"facecolor": PAPER})
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="assets/planning.gif")
    parser.add_argument("--seed", type=int, default=909090)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args(argv)

    frames, start_cm, steps = collect(args.seed, args.steps, args.stride)
    print(render(frames, start_cm, steps, args.out, args.fps), flush=True)


if __name__ == "__main__":
    main()
