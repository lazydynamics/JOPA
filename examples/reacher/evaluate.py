"""Closed-loop evaluation: poses from consecutive seeds, centimetre scoring."""
from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np

from jopa import Agent, Block, JointModel, PoseMotionObservation
from jopa.distributions import gaussian_mean
from jopa.nn.vae import load_params

from .artifacts import (
    ManifestError,
    architecture_for,
    attach_encoded_replay,
    paths,
    sensor_for,
    update_runtime,
    validate_manifest,
    write_json,
)
from .runtime import (
    ACTION_REPEAT,
    CAMERA_DISTANCE,
    EVAL_STEPS,
    GOAL_EIGENVALUE_FLOOR,
    HORIZON,
    IMG_SIZE,
    JOINT2_GOAL_LIMIT,
    JOINT2_START_LIMIT,
    MIN_TASK_SEPARATION,
    MOTION_DIM,
    MOTION_GOAL_PRECISION,
    N_FRAMES,
    NEIGHBORS,
    POSE_DIM,
    POSE_GOAL_PRECISION,
    action_precision,
    latent_subgoal,
)

# A hold counts when the mean fingertip error over the final 20 steps is under
# 3 cm; a pose is settled when it stays there for 95% of the final 60.
HOLD_WINDOW = 20
HOLD_THRESHOLD_CM = 3.0
REACH_THRESHOLD_CM = 1.0
SETTLED_WINDOW = 60
SETTLED_FRACTION = 0.95
SURVEY_POSES = 20
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
    started = time.time()
    artifact = paths(args)
    manifest = validate_manifest(
        artifact["manifest"],
        expected_architecture=architecture_for(args),
        checkpoint_paths={
            "sensor": artifact["sensor"],
            "dynamics": artifact["dynamics"],
            "background": artifact["background"],
        },
    )

    # Simulator packages and coordinates are confined below this line.
    import gymnasium as gym
    import mujoco

    sensor = sensor_for(args)
    sensor_params = load_params(sensor, artifact["sensor"])
    observation = PoseMotionObservation(
        sensor, sensor_params,
        log_std_offsets=manifest["calibration"]["log_std_offsets"])
    with artifact["dynamics"].open("rb") as handle:
        frozen_transition = pickle.load(handle)
    cache_record = manifest.get("encoding_caches", {}).get("train")
    if cache_record is None:
        raise ManifestError(
            "train encoding cache is required for local conjugate refits")
    archive = np.load(cache_record["path"], allow_pickle=False)
    cache_metadata = json.loads(str(archive["metadata"]))
    attach_encoded_replay(
        frozen_transition,
        [{"means": np.asarray(archive[f"means_{i:05d}"]),
          "controls": np.asarray(archive[f"controls_{i:05d}"])}
         for i in range(cache_metadata["trajectories"])],
        NEIGHBORS)
    background = np.load(artifact["background"], allow_pickle=False)
    dimension = POSE_DIM + MOTION_DIM

    environment = gym.make("Reacher-v5", render_mode="rgb_array")
    unwrapped = environment.unwrapped
    world, data = unwrapped.model, unwrapped.data
    fingertip_id = world.geom("fingertip").id
    target_id = world.geom("target").id
    # Render at twice the sensor resolution and reduce with the same 2x2 block
    # mean the training replay went through. Rendering natively at img_size
    # antialiases differently, which shifts the goal latent by a fraction of a
    # pixel — a measurable fraction of a 3 cm budget at 0.43 cm/pixel.
    render_size = 2 * IMG_SIZE
    renderer = mujoco.Renderer(world, render_size, render_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0]
    camera.distance = CAMERA_DISTANCE
    camera.elevation = -90
    camera.azimuth = 90
    target_rgba = world.geom_rgba[target_id].copy()

    def render_rgb(show_target=False):
        world.geom_rgba[target_id] = (
            target_rgba if show_target
            else target_rgba
            * np.array([1, 1, 1, 0], dtype=np.float32))
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
        world.geom_rgba[target_id] = target_rgba
        return image

    def sensor_frame():
        gray = render_rgb(show_target=False).mean(2).astype(np.float32)
        return gray.reshape(
            IMG_SIZE, 2, IMG_SIZE, 2).mean((1, 3)) / 255.0

    def reset(pose, target_xy):
        mujoco.mj_resetData(world, data)
        data.qpos[:2] = pose
        data.qpos[2:4] = target_xy
        data.qvel[:] = 0
        mujoco.mj_forward(world, data)

    def fingertip():
        return data.geom_xpos[fingertip_id][:2].copy()

    def fingertip_of(pose):
        saved_position = data.qpos.copy()
        saved_velocity = data.qvel.copy()
        data.qpos[:2] = pose
        data.qvel[:] = 0
        mujoco.mj_forward(world, data)
        result = fingertip()
        data.qpos[:] = saved_position
        data.qvel[:] = saved_velocity
        mujoco.mj_forward(world, data)
        return result

    def act(control):
        data.ctrl[:] = np.clip(control, -1, 1)
        for _ in range(ACTION_REPEAT):
            mujoco.mj_step(world, data)

    def sample_task(seed):
        rng = np.random.RandomState(seed)
        for _ in range(100):
            start_pose = rng.uniform(-np.pi, np.pi, 2)
            goal_pose = rng.uniform(-np.pi, np.pi, 2)
            start_pose[1] *= JOINT2_START_LIMIT / np.pi
            goal_pose[1] *= JOINT2_GOAL_LIMIT / np.pi
            if np.linalg.norm(
                    fingertip_of(start_pose)
                    - fingertip_of(goal_pose)) > MIN_TASK_SEPARATION:
                return start_pose, goal_pose
        raise RuntimeError("could not construct a separated reachable task")

    def run_pose(label, phase, pose_index, start_pose, goal_pose,
                 pose_seed=None):
        transition = copy.deepcopy(frozen_transition)
        local_model = JointModel([
            Block("z", transition, observe=observation)])
        goal_xy = fingertip_of(goal_pose)
        reset(goal_pose, goal_xy)
        goal_frame = sensor_frame()
        precision = observation.goal_precision(
            goal_frame, background=background,
            pose_mean_precision=POSE_GOAL_PRECISION,
            motion_precision=MOTION_GOAL_PRECISION,
            eigenvalue_floor=GOAL_EIGENVALUE_FLOOR)
        agent = Agent(
            local_model, horizon=HORIZON,
            action_precision=action_precision,
            goal_precision=precision,
            goal_schedule="stage",
            subgoal=latent_subgoal,
            relin_surprise=1e9,
            relin_radius=0.25,
            adapt_hold=False,
            track_uncertainty=True)
        agent.goal(goal_frame)
        goal_latent = np.asarray(observation.encode(goal_frame)[0])
        reset(start_pose, goal_xy)
        history = [sensor_frame()] * N_FRAMES
        trace = {
            "error_cm": [], "latent_error": [], "control_norm": [],
            "observation_std": [], "belief_std": [],
            "action_std": [], "B_std": [], "surprise": [],
        }
        video = []
        for _ in range(EVAL_STEPS):
            control = agent.step(np.stack(history[-N_FRAMES:]))
            act(control)
            history.append(sensor_frame())
            trace["error_cm"].append(float(
                np.linalg.norm(fingertip() - goal_xy) * 100.0))
            trace["latent_error"].append(float(
                np.linalg.norm(
                    np.asarray(gaussian_mean(agent.belief))
                    - goal_latent) / np.sqrt(dimension)))
            trace["control_norm"].append(float(np.linalg.norm(control)))
            trace["observation_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_observation_cov))
                / dimension)))
            trace["belief_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_belief_cov))
                / dimension)))
            trace["action_std"].append(float(np.sqrt(
                np.trace(np.asarray(agent.last_action_cov)) / 2.0)))
            trace["B_std"].append(float(np.mean(
                np.asarray(agent.transition.B_std))))
            trace["surprise"].append(float(agent.surprise))
            if not args.no_video:
                video.append(render_rgb(show_target=True))
        trace = {
            name: np.asarray(values)
            for name, values in trace.items()
        }
        metrics = terminal_metrics(
            trace["error_cm"], HOLD_WINDOW, HOLD_THRESHOLD_CM)
        errors = trace["error_cm"]
        settled_width = min(SETTLED_WINDOW, len(errors))
        tail = errors[-settled_width:]
        metrics.update({
            "phase": phase,
            "pose": int(pose_index),
            "label": label,
            "seed": pose_seed,
            "reached": bool(errors.min() < REACH_THRESHOLD_CM),
            "terminal_20_step_max_cm": float(
                errors[-min(HOLD_WINDOW, len(errors)):].max()),
            "settled": bool(
                float(np.mean(tail < HOLD_THRESHOLD_CM)) >= SETTLED_FRACTION),
        })
        np.savez_compressed(
            artifact["out"] / f"{label}.npz", **trace)
        return {
            "label": label,
            "metrics": metrics,
            "trace": trace,
            "video": video,
        }

    rows = []
    try:
        for index in range(args.poses):
            pose_seed = args.pose_seed + index
            start_pose, goal_pose = sample_task(pose_seed)
            run = run_pose(
                f"survey_{args.pose_seed}_{index:02d}", "survey", index,
                start_pose, goal_pose, pose_seed=pose_seed)
            rows.append(run["metrics"])
            print(
                f"  pose {index:02d} seed {pose_seed}  "
                f"min {run['metrics']['minimum_cm']:.2f} cm  "
                f"terminal {run['metrics']['terminal_20_step_cm']:.2f} cm",
                flush=True)
    finally:
        renderer.close()
        environment.close()
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


