"""Closed-loop evaluation: poses from consecutive seeds, centimetre scoring."""
from __future__ import annotations

import json
import time

import numpy as np

from jopa.distributions import gaussian_mean

from .artifacts import (
    architecture_for,
    load_manifest,
    paths,
    update_runtime,
    validate_manifest,
    write_json,
)
from .runtime import EVAL_STEPS, N_FRAMES, ReacherLoop

# A hold counts when the mean fingertip error over the final 20 steps is under
# 3 cm; a pose is settled when it stays there for 95% of the final 60.
HOLD_WINDOW = 20
HOLD_THRESHOLD_CM = 3.0
REACH_THRESHOLD_CM = 1.0
SETTLED_WINDOW = 60
SETTLED_FRACTION = 0.95
SURVEY_POSES = 20
CLIP_FPS = 25
DEFAULT_POSE_SEED = 515151
MONTAGE_FPS = 25

def longest_run(values, threshold):
    best = current = 0
    for value in values:
        current = current + 1 if value < threshold else 0
        best = max(best, current)
    return int(best)


def terminal_metrics(errors, window, threshold):
    errors = np.asarray(errors)
    width = min(int(window), len(errors))
    rolling = np.convolve(
        errors, np.ones(width) / width, mode="valid")
    best_index = int(np.argmin(rolling))
    terminal = float(np.mean(errors[-width:]))
    return {
        "minimum_cm": float(errors.min()),
        "best_20_step_cm": float(rolling[best_index]),
        "best_20_step_start": best_index,
        "terminal_20_step_cm": terminal,
        "longest_run_below_3cm": longest_run(errors, threshold),
        "terminal_success": bool(terminal < threshold),
    }


def run_evaluate(args):
    """Closed-loop evaluation. Drives `ReacherLoop`, which is the single place
    the simulator, the frozen artifacts and the agent settings are defined —
    so evaluation and the figure scripts cannot diverge."""
    started = time.time()
    artifact = paths(args)
    validate_manifest(
        load_manifest(artifact["manifest"]),
        expected_architecture=architecture_for(args),
        checkpoint_paths={
            "sensor": artifact["sensor"],
            "dynamics": artifact["dynamics"],
            "background": artifact["background"],
        },
    )
    loop = ReacherLoop(artifact["out"], record=not args.no_video)

    def run_pose(label, pose_index, start_pose, goal_pose, pose_seed=None):
        goal_xy = loop.fingertip_of(goal_pose)
        loop.reset(goal_pose, goal_xy)
        goal_frame = loop.sensor_frame()
        agent = loop.agent_for(goal_frame, track_uncertainty=True)
        goal_latent = np.asarray(loop.observation.encode(goal_frame)[0])
        loop.reset(start_pose, goal_xy)
        history = [loop.sensor_frame()] * N_FRAMES
        trace = {
            "error_cm": [], "latent_error": [], "control_norm": [],
            "observation_std": [], "belief_std": [], "action_std": [],
            "B_std": [], "surprise": [], "drift_norm": [], "drift_std": [],
        }
        video = []
        for _ in range(EVAL_STEPS):
            control = agent.step(np.stack(history[-N_FRAMES:]))
            loop.act(control)
            history.append(loop.sensor_frame())
            trace["error_cm"].append(float(
                np.linalg.norm(loop.fingertip() - goal_xy) * 100.0))
            trace["latent_error"].append(float(np.linalg.norm(
                np.asarray(gaussian_mean(agent.belief)) - goal_latent)))
            trace["control_norm"].append(float(np.linalg.norm(control)))
            trace["observation_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_observation_cov)))))
            trace["belief_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_belief_cov)))))
            trace["action_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_action_cov)) / 2.0)))
            trace["B_std"].append(float(np.mean(
                np.asarray(agent.transition.B_std))))
            trace["surprise"].append(float(agent.surprise))
            drift, drift_std = agent.transition.c, agent.transition.c_std
            if drift is not None:
                trace["drift_norm"].append(float(np.linalg.norm(drift)))
                trace["drift_std"].append(float(np.mean(drift_std)))
            if not args.no_video:
                video.append(loop.display_frame())
        trace = {name: np.asarray(values) for name, values in trace.items()}
        metrics = terminal_metrics(
            trace["error_cm"], HOLD_WINDOW, HOLD_THRESHOLD_CM)
        errors = trace["error_cm"]
        tail_width = min(SETTLED_WINDOW, len(errors))
        metrics.update({
            "pose": int(pose_index),
            "label": label,
            "seed": pose_seed,
            "reached": bool(errors.min() < REACH_THRESHOLD_CM),
            "terminal_20_step_max_cm": float(
                errors[-min(HOLD_WINDOW, len(errors)):].max()),
            "settled": bool(float(np.mean(
                errors[-tail_width:] < HOLD_THRESHOLD_CM)) >= SETTLED_FRACTION),
        })
        np.savez_compressed(artifact["out"] / f"{label}.npz", **trace)
        if video:
            import imageio.v2 as imageio
            imageio.mimsave(
                artifact["out"] / f"{label}.gif", video, fps=CLIP_FPS, loop=0)
        return {"label": label, "metrics": metrics}

    rows = []
    try:
        for index in range(args.poses):
            pose_seed = args.pose_seed + index
            start_pose, goal_pose = loop.sample_task(pose_seed)
            run = run_pose(
                f"survey_{args.pose_seed}_{index:02d}", index,
                start_pose, goal_pose, pose_seed=pose_seed)
            rows.append(run["metrics"])
            print(
                f"  pose {index:02d} seed {pose_seed}  "
                f"min {run['metrics']['minimum_cm']:.2f} cm  "
                f"terminal {run['metrics']['terminal_20_step_cm']:.2f} cm",
                flush=True)
    finally:
        loop.close()
    summary = {
        "pose_seed": int(args.pose_seed),
        "poses": len(rows),
        "steps": int(EVAL_STEPS),
        "reached_under_1cm": sum(row["reached"] for row in rows),
        "terminal_mean_under_3cm": sum(
            row["terminal_success"] for row in rows),
        "terminal_max_under_3cm": sum(
            row["terminal_20_step_max_cm"] < HOLD_THRESHOLD_CM
            for row in rows),
        "settled": sum(row["settled"] for row in rows),
    }
    write_json(
        artifact["out"] / f"survey_{args.pose_seed}.json",
        {"summary": summary, "rows": rows})
    update_runtime(
        artifact["runtime"], "evaluate", time.time() - started,
        result="survey", pose_seed=int(args.pose_seed),
        poses=len(rows))
    print(json.dumps(summary, indent=2), flush=True)
    return


