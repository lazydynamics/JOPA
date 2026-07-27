"""Square card for a post: the graph on top, what it decides underneath.

The message sweeps the chain once over the clip while the arm closes on the
goal, so the two halves read as one operation rather than two systems.

    MUJOCO_GL=egl python figures/post_card.py --out previews/post_card.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jopa.distributions import gaussian_mean_cov  # noqa: E402

from examples.reacher.runtime import ReacherLoop  # noqa: E402
from style import (  # noqa: E402
    ACCENT, INK, INK_LIGHT, PAPER, RULE, SLATE, _bare_axes, _draw_graph,
    _layout, _schedule,
)


def rolled_out(agent, goal_latent, pose_dim=4):
    """Belief pushed forward under the plan: distance to goal, with spread.

    The spread is the belief covariance projected on the direction of the
    remaining error, so it is the model's own uncertainty about how far it
    still has to go.
    """
    plan = np.asarray(agent.last_plan.mean)[:, :agent.du]
    belief = agent.belief
    means, spreads = [], []
    for step in range(len(plan) + 1):
        mean, covariance = gaussian_mean_cov(belief)
        mean, covariance = np.asarray(mean), np.asarray(covariance)
        delta = mean[:pose_dim] - goal_latent[:pose_dim]
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        block = covariance[:pose_dim, :pose_dim]
        means.append(distance)
        spreads.append(float(np.sqrt(max(direction @ block @ direction, 0.0))))
        if step < len(plan):
            belief = agent.transition.predict(belief, plan[step])
    return np.array(means), np.array(spreads)


def collect(seed, steps, stride):
    loop = ReacherLoop(record=True)
    start_pose, goal_pose = loop.sample_task(seed)
    goal_xy = loop.fingertip_of(goal_pose)
    loop.reset(goal_pose, goal_xy)
    goal_frame = loop.sensor_frame()
    goal_latent = np.asarray(loop.observation.encode(goal_frame)[0])
    agent = loop.agent_for(goal_frame, track_uncertainty=True)
    loop.reset(start_pose, goal_xy)
    history = [loop.sensor_frame()] * 4

    frames, error = [], []
    for step in range(steps):
        action = agent.step(np.stack(history[-4:]))
        posterior = agent.last_plan
        torque = np.asarray(posterior.mean)[:, :agent.du]
        torque_spread = np.sqrt(np.diagonal(
            np.asarray(posterior.marginal_covariance),
            axis1=1, axis2=2))[:, :agent.du]
        predicted, spread = rolled_out(agent, goal_latent)
        image = loop.display_frame()
        loop.act(action)
        history.append(loop.sensor_frame())
        error.append(float(np.linalg.norm(loop.fingertip() - goal_xy) * 100.0))
        if step % stride == 0:
            frames.append({
                "image": image, "predicted": predicted, "spread": spread,
                "torque": torque, "torque_spread": torque_spread,
                "error": list(error), "step": step,
            })
    loop.close()
    return frames


def render(frames, path, fps, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    layout = _layout()
    schedule = _schedule(layout)
    horizon = len(frames[0]["predicted"])
    channels = frames[0]["torque"].shape[1]
    # A single uncertain frame would otherwise flatten every confident one.
    ceiling = float(np.quantile(
        [(f["predicted"] + 2 * f["spread"]).max() for f in frames], 0.85))
    reach = max((np.abs(f["torque"]) + f["torque_spread"]).max()
                for f in frames)

    figure = plt.figure(figsize=(9.0, 9.0), facecolor=PAPER)
    ax_graph = figure.add_axes([0.02, 0.555, 0.96, 0.325])
    ax_arm = figure.add_axes([0.045, 0.075, 0.375, 0.375])
    ax_state = figure.add_axes([0.590, 0.300, 0.370, 0.175])
    ax_plan = figure.add_axes([0.590, 0.075, 0.370, 0.150])

    figure.text(0.5, 0.977, "Joint Observation–Planning Architecture",
                color=INK, fontsize=15, ha="center", va="top")
    figure.text(0.5, 0.941,
                "one graph: where it is, and what to do about it",
                color=INK_LIGHT, fontsize=11, ha="center", va="top")
    figure.text(0.5, 0.516,
                "the sweep that locates the arm is the sweep that plans the "
                "torque",
                color=INK_LIGHT, fontsize=10.5, ha="center", va="top")
    figure.text(0.960, 0.487, "where it believes it will be, ±2σ",
                color=INK_LIGHT, fontsize=9, ha="right", va="top")

    def update(index):
        frame = frames[index]
        position = index / max(len(frames) - 1, 1) * len(schedule)
        hop = min(int(position), len(schedule) - 1)
        start, end, _ = schedule[hop]
        _draw_graph(ax_graph, layout, active=(start, end, position - hop),
                    caption="", annotate=False)

        ax_arm.clear()
        ax_arm.axis("off")
        ax_arm.imshow(frame["image"])
        ax_arm.text(0.5, -0.04,
                    f"{frame['error'][-1]:.2f} cm to goal   ·   "
                    f"step {frame['step']}",
                    transform=ax_arm.transAxes, color=INK, fontsize=10,
                    ha="center", va="top")

        _bare_axes(ax_state)
        ahead = np.arange(horizon)
        predicted, spread = frame["predicted"], frame["spread"]
        ax_state.fill_between(ahead, predicted - 2 * spread,
                              predicted + 2 * spread, color=SLATE, alpha=0.24,
                              linewidth=0)
        ax_state.plot(ahead, predicted, color=INK, lw=1.8)
        ax_state.plot([0], [predicted[0]], marker="o", ms=6, color=ACCENT)
        ax_state.set_xlim(0, horizon - 1)
        ax_state.set_ylim(0, ceiling * 1.06)
        ax_state.set_xticks(list(ahead))
        ax_state.set_ylabel("distance to goal", color=INK_LIGHT, fontsize=9.5)


        _bare_axes(ax_plan)
        torque, torque_spread = frame["torque"], frame["torque_spread"]
        offsets = np.linspace(-0.16, 0.16, channels)
        ax_plan.axhline(0.0, color=RULE, lw=1.0)
        for channel in range(channels):
            for step in range(len(torque)):
                colour = ACCENT if step == 0 else SLATE
                at = step + offsets[channel]
                ax_plan.vlines(
                    at, torque[step, channel] - torque_spread[step, channel],
                    torque[step, channel] + torque_spread[step, channel],
                    color=colour, lw=1.3, alpha=0.5)
                ax_plan.plot([at], [torque[step, channel]],
                             marker="o" if channel == 0 else "s", ms=4.0,
                             color=colour)
        ax_plan.set_xlim(-0.5, len(torque) - 0.5)
        ax_plan.set_ylim(-reach * 1.25, reach * 1.25)
        ax_plan.set_xticks(range(len(torque)))
        ax_plan.set_xlabel("steps ahead", color=INK_LIGHT, fontsize=9.5)
        ax_plan.set_ylabel("planned torque", color=INK_LIGHT, fontsize=9.5)
        ax_plan.text(len(torque) - 1, -reach * 1.12,
                     "first applied, rest re-inferred", color=INK_LIGHT,
                     fontsize=9, ha="right", va="bottom")
        return []

    animation = FuncAnimation(figure, update, frames=len(frames), blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi,
                   savefig_kwargs={"facecolor": PAPER})
    plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="previews/post_card.gif")
    parser.add_argument("--seed", type=int, default=909090)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args(argv)

    print(render(collect(args.seed, args.steps, args.stride), args.out,
                 args.fps, args.dpi), flush=True)


if __name__ == "__main__":
    main()
