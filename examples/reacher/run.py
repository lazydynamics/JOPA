"""Canonical pixel-only structured-state MuJoCo Reacher audit.

Usage:
    python -m examples.reacher.run train --data outputs/gpu_run
    python -m examples.reacher.run validate --data outputs/gpu_run
    MUJOCO_GL=egl python -m examples.reacher.run evaluate

Training and admission use only agent-generated grayscale pixels and actions.
MuJoCo coordinates are imported only by evaluate, where they construct
reachable tasks and compute the final centimetre audit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jopa import (
    Agent,
    Block,
    Gaussian,
    JointModel,
    LearnedLinear,
    PoseMotionObservation,
)
from jopa.distributions import gaussian_mean
from jopa.nn.vae import (
    PoseMotionVAE,
    load_params,
    save_params,
    train_pose_motion_vae,
)

from .runtime import (
    ACTION_PRECISION_HOLD,
    ACTION_PRECISION_TRAVEL,
    ACTION_PRIOR_POWER,
    ACTION_PRIOR_SCALE,
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
    MOTION_HIDDEN_DIM,
    N_FRAMES,
    NEIGHBORS,
    POSE_DIM,
    POSE_GOAL_PRECISION,
    REFIT_OBS_PRECISION,
    SENSOR_CHANNELS,
    SUBGOAL_TRUST,
    action_precision,
    latent_subgoal,
)
from .validation import (
    FAILED_AUDIT_SEED,
    MANIFEST_SCHEMA,
    SMOKE_SEED,
    ManifestError,
    admission_gate,
    checkpoint_sha256,
    innovation_interval_coverage,
    load_manifest,
    pixel_action_innovations,
    repeated_frame_motion_rms,
    rollout_diagnostics,
    split_trajectories,
    validate_manifest,
    visual_aliasing_metrics,
    write_manifest,
    zero_action_fixed_point_drift,
)

# Fixed at their validated values by the recorded protocol run; the planning
# and sensor geometry constants they pair with live in ``runtime`` so the
# evaluation loop and the figure scripts cannot drift apart.
REPLAY_PATTERN = "ck_*.pkl"
PREDICTION_WEIGHT = 10.0
VMP_ITERATIONS = 12
SENSOR_SEED = 8
SPLIT_SEED = 41
VALIDATION_REPEATS = 2
# Flat action precision, superseded by the conditional prior in ``runtime``
# and recorded in the manifest for provenance.
ACTION_PRECISION = 1.0
REFIT_REFRESH_DISTANCE = 0.0
# Physical audit protocol: a hold counts when the mean fingertip error over the
# final 20 closed-loop steps stays under 3 cm.
HOLD_WINDOW = 20
HOLD_THRESHOLD_CM = 3.0
GATE_B_SEED = 73108
MONTAGE_FPS = 25


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
    return parser


def paths(args):
    output = Path(args.outdir)
    return {
        "out": output,
        "manifest": output / "manifest.json",
        "sensor": output / "pose_motion_sensor.msgpack",
        "bootstrap_dynamics": output / "conjugate_dynamics_bootstrap.pkl",
        "dynamics": output / "conjugate_dynamics.pkl",
        "background": output / "background.npy",
        "sensor_diagnostics": output / "sensor_diagnostics.json",
        "runtime": output / "runtime.json",
        "failed_audit": output / f"audit_seed_{FAILED_AUDIT_SEED}.json",
    }


def sensor_for(args):
    return PoseMotionVAE(
        pose_dim=POSE_DIM,
        motion_dim=MOTION_DIM,
        ch=args.channels,
        n_frames=N_FRAMES,
        img_size=IMG_SIZE,
        motion_hidden_dim=MOTION_HIDDEN_DIM,
    )


def architecture_for(args):
    return sensor_for(args).architecture()


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "shape"):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n")


def update_runtime(path, mode, seconds, **extra):
    report = {}
    if Path(path).is_file():
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            report = {}
    report[mode] = {
        "seconds": float(seconds),
        **json_value(extra),
    }
    write_json(path, report)


def load_pixel_action_replay(args):
    names = sorted(
        path.name for path in Path(args.data).glob(REPLAY_PATTERN))
    usable = []
    provenance = []
    for name in names:
        source = Path(args.data) / name
        if not source.is_file():
            continue
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        accepted = 0
        for chunk in loaded:
            # Only tuple fields zero and one are authorized. A legacy third
            # simulator-state field is deliberately never indexed.
            frames = np.asarray(chunk[0])
            controls = np.asarray(chunk[1])
            if frames.ndim != 3 or controls.ndim != 2:
                continue
            if len(frames) != len(controls) + 1:
                continue
            if len(frames) < N_FRAMES + args.rollout:
                continue
            if frames.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                old_height, old_width = frames.shape[-2:]
                if old_height != old_width or old_height % IMG_SIZE:
                    raise ValueError(
                        f"cannot downsample {frames.shape[-2:]} "
                        f"to {(IMG_SIZE, IMG_SIZE)}")
                scale = old_height // IMG_SIZE
                frames = frames.reshape(
                    len(frames), IMG_SIZE, scale,
                    IMG_SIZE, scale).mean((2, 4))
            frames = frames.astype(np.float32)
            if frames.max(initial=0.0) > 1.0:
                frames /= 255.0
            usable.append((frames, controls.astype(np.float32)))
            accepted += 1
        provenance.append({
            "path": str(source.resolve()),
            "sha256": checkpoint_sha256(source),
            "loaded_chunks": len(loaded),
            "accepted_chunks": int(accepted),
        })
    if len(usable) < 3:
        raise ValueError(
            f"found only {len(usable)} usable pixel/action trajectories")
    return usable, provenance


def split_replay(args, manifest=None):
    replay, provenance = load_pixel_action_replay(args)
    split = split_trajectories(replay, seed=SPLIT_SEED)
    if manifest is not None:
        expected = manifest["split"]["indices"]
        if split["indices"] != expected:
            raise ManifestError(
                "current replay does not reproduce manifest split indices")
        if manifest.get("replay_provenance") != provenance:
            raise ManifestError(
                "replay files differ from the frozen manifest")
    return replay, split, provenance


def cache_path(output, name, sensor_hash):
    return output / f"encodings_{name}_{sensor_hash[:16]}.npz"


def encode_trajectory(observation, trajectory, batch_size=512):
    frames, controls = trajectory
    count = len(frames) - observation.n_frames + 1
    windows = np.stack([
        frames[index:index + observation.n_frames]
        for index in range(count)])
    means = []
    log_stds = []
    for start in range(0, count, batch_size):
        mean, log_std = observation.encode_batch(
            windows[start:start + batch_size])
        means.append(np.asarray(mean))
        log_stds.append(np.asarray(log_std))
    return {
        "means": np.concatenate(means),
        "log_stds": np.concatenate(log_stds),
        "controls": controls[observation.n_frames - 1:],
        "images": frames[observation.n_frames - 1:],
    }


def save_encoding_cache(path, encoded, metadata):
    arrays = {"metadata": np.asarray(json.dumps(metadata))}
    for index, item in enumerate(encoded):
        for name in ("means", "log_stds", "controls", "images"):
            arrays[f"{name}_{index:05d}"] = item[name]
    np.savez_compressed(path, **arrays)


def load_encoding_cache(path, expected_metadata):
    if not Path(path).is_file():
        return None
    try:
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        if metadata != expected_metadata:
            return None
        encoded = []
        for index in range(metadata["trajectories"]):
            encoded.append({
                name: np.asarray(archive[f"{name}_{index:05d}"])
                for name in ("means", "log_stds", "controls", "images")
            })
        return encoded
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def encodings_for(
        args, split, observation, sensor_hash, *, force=False):
    result = {}
    for name in ("train", "calibration", "test"):
        metadata = {
            "sensor_sha256": sensor_hash,
            "split": name,
            "indices": split["indices"][name],
            "trajectories": len(split[name]),
            "calibrated": False,
        }
        path = cache_path(Path(args.outdir), name, sensor_hash)
        encoded = (
            None if force
            else load_encoding_cache(path, metadata))
        if encoded is None:
            encoded = [
                encode_trajectory(observation, trajectory)
                for trajectory in split[name]]
            save_encoding_cache(path, encoded, metadata)
        result[name] = encoded
    return result


def gaussian_sequence(item, offsets):
    messages = []
    for mean, log_std in zip(item["means"], item["log_stds"]):
        calibrated = np.clip(
            log_std + offsets, -6.0, 2.0)
        precision = jnp.diag(jnp.exp(-2.0 * jnp.asarray(calibrated)))
        mean_array = jnp.asarray(mean)
        messages.append(Gaussian(
            eta=precision @ mean_array, lam=precision))
    return messages


def fit_conjugate_dynamics(encoded, offsets):
    dimension = POSE_DIM + MOTION_DIM
    transition = LearnedLinear(
        dimension, du=2, offset=True,
        n_iterations=VMP_ITERATIONS,
        prior_a_cov=0.5, init_a_cov=0.5,
        prior_b_cov=100.0, init_b_cov=100.0,
        prior_W_df=dimension + 2)
    transition.learn(
        [gaussian_sequence(item, offsets) for item in encoded],
        [item["controls"] for item in encoded])
    return transition


def checkpoint_record(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": checkpoint_sha256(path),
    }


def training_configuration(args):
    return {
        "pose_steps": int(args.pose_steps),
        "controlled_steps": int(args.controlled_steps),
        "rollout": int(args.rollout),
        "prediction_weight": float(PREDICTION_WEIGHT),
        "batch_size": int(args.batch),
        "sensor_seed": int(SENSOR_SEED),
        "vmp_iterations": int(VMP_ITERATIONS),
        "pixels_and_actions_only": True,
        "bootstrap_predictor_deployed": False,
    }


def base_manifest(args, split, provenance):
    return {
        "schema": MANIFEST_SCHEMA,
        "phase": "initializing",
        "architecture": architecture_for(args),
        "training": training_configuration(args),
        "split": {
            "seed": int(split["seed"]),
            "ratios": split["ratios"],
            "indices": split["indices"],
        },
        "replay_provenance": provenance,
        "checkpoints": {},
        "calibration": None,
        "admission": None,
        "reproducibility": None,
        "scientific_policy": {
            "sensor_input": ["pixels", "actions"],
            "simulator_labels_read_during_training": False,
            "simulator_labels_read_during_validation": False,
            "bootstrap_linear_predictor_disposable": True,
            "deployed_dynamics": (
                "JOPA conjugate q(A,B,W), refit locally at the belief "
                "from frozen pixel-latent replay"),
            "planner": "infer_actions_exact",
            "amended_planning": {
                "pose_goal_precision": float(POSE_GOAL_PRECISION),
                "motion_goal_precision": float(MOTION_GOAL_PRECISION),
                "action_precision": float(ACTION_PRECISION),
                "action_prior": {
                    "kind": "conditional flat-top on latent goal distance",
                    "travel_precision": float(ACTION_PRECISION_TRAVEL),
                    "hold_precision": float(ACTION_PRECISION_HOLD),
                    "scale": float(ACTION_PRIOR_SCALE),
                    "power": float(ACTION_PRIOR_POWER),
                },
                "subgoal": "latent trust region",
                "subgoal_trust": float(SUBGOAL_TRUST),
                "localize_neighbors": int(NEIGHBORS),
                "refit_obs_prec": float(REFIT_OBS_PRECISION),
                "rollout_gate_localized": True,
                "rationale": (
                    "oracle diagnostics showed a single global linear model "
                    "has vanishing velocity-to-pose coupling on angle "
                    "embeddings and motion-heavy goal precision suppresses "
                    "movement; state was used only on the diagnostic bench, "
                    "never in the deployed loop"),
            },
        },
        "physical_audit": {
            "preserved_failed_seed": FAILED_AUDIT_SEED,
            "failed_seed_used_for_tuning": False,
            "smoke_seed": SMOKE_SEED,
            "gate_b_seed": int(GATE_B_SEED),
            "gate_b_required_successes": 7,
            "gate_b_poses": 8,
        },
    }


def plot_training_curves(diagnostics, path):
    import matplotlib.pyplot as plt

    history = diagnostics.get("history", {})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    pose = history.get("pose", [])
    controlled = history.get("controlled", [])
    if pose:
        axes[0].plot(
            [point["step"] for point in pose],
            [point["loss"] for point in pose], label="pose total")
        axes[0].plot(
            [point["step"] for point in pose],
            [point["visual_distance"] for point in pose],
            label="visual distance")
    if controlled:
        axes[1].plot(
            [point["step"] for point in controlled],
            [point["loss"] for point in controlled], label="total")
        axes[1].plot(
            [point["step"] for point in controlled],
            [point["prediction"] for point in controlled],
            label="12-step training prediction")
        axes[1].plot(
            [point["step"] for point in controlled],
            [point["zero_motion"] for point in controlled],
            label="repeated-frame motion")
    for axis, title in zip(
            axes, ("pose pretraining", "controlled training")):
        axis.set_title(title)
        axis.set_xlabel("optimizer step")
        axis.set_yscale("symlog")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_train(args):
    started = time.time()
    artifact = paths(args)
    artifact["out"].mkdir(parents=True, exist_ok=True)
    _, split, provenance = split_replay(args)
    manifest = base_manifest(args, split, provenance)
    if artifact["manifest"].is_file() and not args.retrain:
        previous = load_manifest(artifact["manifest"])
        if previous.get("architecture") != manifest["architecture"]:
            raise ManifestError(
                "existing manifest architecture is incompatible; "
                "use a new outdir or --retrain")
        if previous.get("split") != manifest["split"]:
            raise ManifestError(
                "existing manifest uses a different trajectory split")
        manifest = previous
    write_manifest(artifact["manifest"], manifest)

    sensor = sensor_for(args)
    if artifact["sensor"].is_file() and not args.retrain:
        sensor_params = load_params(sensor, artifact["sensor"])
        diagnostics = (
            json.loads(artifact["sensor_diagnostics"].read_text())
            if artifact["sensor_diagnostics"].is_file() else {})
    else:
        sensor, sensor_params, diagnostics = train_pose_motion_vae(
            [item[0] for item in split["train"]],
            [item[1] for item in split["train"]],
            pose_dim=POSE_DIM,
            motion_dim=MOTION_DIM,
            n_frames=N_FRAMES,
            rollout=args.rollout,
            ch=args.channels,
            motion_hidden_dim=MOTION_HIDDEN_DIM,
            img_size=IMG_SIZE,
            pose_steps=args.pose_steps,
            controlled_steps=args.controlled_steps,
            batch_size=args.batch,
            prediction_weight=PREDICTION_WEIGHT,
            seed=SENSOR_SEED)
        save_params(sensor_params, artifact["sensor"])
        write_json(artifact["sensor_diagnostics"], diagnostics)
        plot_training_curves(
            json_value(diagnostics),
            artifact["out"] / "training_curves.png")
    sensor_hash = checkpoint_sha256(artifact["sensor"])
    manifest["checkpoints"]["sensor"] = checkpoint_record(
        artifact["sensor"])
    manifest["phase"] = "sensor_trained"
    write_manifest(artifact["manifest"], manifest)

    observation = PoseMotionObservation(sensor, sensor_params)
    encoded = encodings_for(
        args, split, observation, sensor_hash, force=args.retrain)
    background = np.median(np.concatenate([
        item[0][
            np.linspace(0, len(item[0]) - 1, min(8, len(item[0])),
                        dtype=int)]
        for item in split["train"]
    ]), axis=0).astype(np.float32)
    np.save(artifact["background"], background)

    if artifact["bootstrap_dynamics"].is_file() and not args.retrain:
        with artifact["bootstrap_dynamics"].open("rb") as handle:
            transition = pickle.load(handle)
    else:
        transition = fit_conjugate_dynamics(
            encoded["train"], np.zeros(POSE_DIM + MOTION_DIM))
        with artifact["bootstrap_dynamics"].open("wb") as handle:
            pickle.dump(transition, handle)
    with artifact["dynamics"].open("wb") as handle:
        pickle.dump(transition, handle)
    manifest["checkpoints"]["bootstrap_dynamics"] = checkpoint_record(
        artifact["bootstrap_dynamics"])
    manifest["checkpoints"]["dynamics"] = checkpoint_record(
        artifact["dynamics"])
    manifest["checkpoints"]["background"] = checkpoint_record(
        artifact["background"])
    manifest["encoding_caches"] = {
        name: checkpoint_record(cache_path(
            artifact["out"], name, sensor_hash))
        for name in ("train", "calibration", "test")
    }
    manifest["phase"] = "trained"
    manifest["admission"] = None
    manifest["reproducibility"] = None
    write_manifest(artifact["manifest"], manifest)
    if not artifact["failed_audit"].exists():
        write_json(artifact["failed_audit"], {
            "seed": FAILED_AUDIT_SEED,
            "status": "failed audit preserved by protocol",
            "used_for_hyperparameter_selection": False,
            "rerun_by_this_pipeline": False,
        })
    update_runtime(
        artifact["runtime"], "train", time.time() - started,
        jax_backend=jax.default_backend(),
        trajectories={
            name: len(split[name])
            for name in ("train", "calibration", "test")
        })
    print(
        f"trained structured sensor and conjugate dynamics: "
        f"{artifact['manifest']}", flush=True)


def encoded_tuples(encoded):
    return [
        (item["means"], item["controls"])
        for item in encoded
    ]


def concatenate(encoded, name):
    return np.concatenate([item[name] for item in encoded], axis=0)


def static_states(observation, images, batch_size=512):
    result = []
    for start in range(0, len(images), batch_size):
        mean, _ = observation.encode_goal_batch(
            images[start:start + batch_size])
        result.append(np.asarray(mean))
    return np.concatenate(result)


def attach_encoded_replay(transition, encoded, neighbors,
                          obs_prec=REFIT_OBS_PRECISION,
                          refresh_distance=REFIT_REFRESH_DISTANCE):
    transition.attach_replay(
        [item["means"] for item in encoded],
        [item["controls"] for item in encoded],
        neighbors=neighbors, refresh_every=1, obs_prec=obs_prec,
        refresh_distance=refresh_distance)
    return transition


def validation_report(
        transition, observation, encoded_test, background,
        replay_encoded=None, neighbors=NEIGHBORS,
        refit_obs_prec=REFIT_OBS_PRECISION):
    # The deployed model is the conjugate posterior refit locally around the
    # current belief from frozen pixel-latent replay; the rollout gate
    # evaluates that model. A copy keeps the global posterior — used by the
    # innovation and drift checks below — untouched by localization.
    if replay_encoded is not None:
        local = attach_encoded_replay(
            copy.deepcopy(transition), replay_encoded, neighbors,
            obs_prec=refit_obs_prec)
    else:
        local = transition
    rollout = rollout_diagnostics(
        local, encoded_tuples(encoded_test),
        horizon=HORIZON, samples=None, localize=replay_encoded is not None)
    states = concatenate(encoded_test, "means")
    images = concatenate(encoded_test, "images")
    visual = visual_aliasing_metrics(
        images, states, pose_dim=observation.pose_dim,
        background=background, seed=0)
    rest = static_states(observation, images)
    rest_rms = repeated_frame_motion_rms(
        rest, states, pose_dim=observation.pose_dim)
    drift = zero_action_fixed_point_drift(
        transition, rest, states, steps=20)
    innovations, base_log_stds = pixel_action_innovations(
        transition, encoded_test)
    calibrated_logs = np.clip(
        base_log_stds + np.asarray(observation.log_std_offsets),
        -6.0, 2.0)
    coverage = innovation_interval_coverage(
        innovations, log_stds=calibrated_logs, level=0.90)
    report = {
        **rollout,
        **visual,
        "repeated_frame_motion_rms": float(rest_rms),
        "zero_action_drift_20_nrmse": float(drift),
        "innovation_90_coverage": float(coverage),
        "test_trajectories": len(encoded_test),
        "test_states": len(states),
    }
    gate = admission_gate(report)
    return report, gate


def run_validate(args):
    started = time.time()
    artifact = paths(args)
    manifest = load_manifest(artifact["manifest"])
    validate_manifest(
        manifest,
        expected_architecture=architecture_for(args),
        checkpoint_paths={
            "sensor": artifact["sensor"],
            "dynamics": artifact["dynamics"],
            "background": artifact["background"],
        },
        require_passed=False)
    _, split, _ = split_replay(args, manifest)
    sensor = sensor_for(args)
    sensor_params = load_params(sensor, artifact["sensor"])
    uncalibrated = PoseMotionObservation(sensor, sensor_params)
    sensor_hash = checkpoint_sha256(artifact["sensor"])
    encoded = encodings_for(
        args, split, uncalibrated, sensor_hash)

    with artifact["bootstrap_dynamics"].open("rb") as handle:
        transition = pickle.load(handle)
    offsets = np.zeros(POSE_DIM + MOTION_DIM, dtype=np.float32)
    # Two deterministic calibration/refit rounds use calibration replay only.
    for _ in range(2):
        innovations, base_logs = pixel_action_innovations(
            transition, encoded["calibration"])
        offsets = PoseMotionObservation.calibration_offsets(
            innovations, base_logs, coverage=0.90)
        transition = fit_conjugate_dynamics(encoded["train"], offsets)
    with artifact["dynamics"].open("wb") as handle:
        pickle.dump(transition, handle)

    observation = PoseMotionObservation(
        sensor, sensor_params, log_std_offsets=offsets)
    background = np.load(artifact["background"], allow_pickle=False)
    repeats = max(2, VALIDATION_REPEATS)
    reports = []
    gates = []
    for _ in range(repeats):
        report, gate = validation_report(
            transition, observation, encoded["test"], background,
            replay_encoded=encoded["train"])
        reports.append(report)
        gates.append(gate)
    canonical = json.dumps(
        {"report": reports[0], "gate": gates[0]},
        sort_keys=True, separators=(",", ":"))
    reproducible = all(
        json.dumps(
            {"report": report, "gate": gate},
            sort_keys=True, separators=(",", ":")) == canonical
        for report, gate in zip(reports[1:], gates[1:]))

    manifest["checkpoints"]["dynamics"] = checkpoint_record(
        artifact["dynamics"])
    manifest["calibration"] = {
        "source_split": "calibration",
        "method": "per-dimension 90% innovation quantile matching",
        "log_std_offsets": offsets.tolist(),
        "target_coverage": 0.90,
        "simulator_labels_used": False,
    }
    manifest["admission"] = {
        **gates[0],
        "report": reports[0],
        "source_split": "test",
        "untouched_during_training_and_calibration": True,
    }
    manifest["reproducibility"] = {
        "validation_runs": int(repeats),
        "identical": bool(reproducible),
        "report_digest": __import__("hashlib").sha256(
            canonical.encode("utf-8")).hexdigest(),
    }
    manifest["phase"] = (
        "validated" if gates[0]["passed"] and reproducible
        else "validation_failed")
    write_manifest(artifact["manifest"], manifest)
    write_json(artifact["out"] / "admission_report.json", {
        "report": reports[0],
        "admission": gates[0],
        "reproducibility": manifest["reproducibility"],
    })
    update_runtime(
        artifact["runtime"], "validate", time.time() - started,
        validation_runs=repeats,
        admission_passed=gates[0]["passed"],
        reproducible=reproducible)
    print(json.dumps(manifest["admission"], indent=2), flush=True)
    if not gates[0]["passed"]:
        raise RuntimeError(
            "state-free admission gates failed; MuJoCo evaluation is forbidden")
    if not reproducible:
        raise RuntimeError(
            "validation was not reproducible; MuJoCo evaluation is forbidden")


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


def save_trace_plot(output, label, trace, metrics, threshold):
    import matplotlib.pyplot as plt

    steps = np.arange(len(trace["error_cm"]))
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, trace["error_cm"], color="tab:red")
    axes[0].axhline(threshold, color="black", linestyle="--")
    axes[0].set_ylabel("error (cm)")
    axes[0].set_title(
        f"{label}: terminal 20-step "
        f"{metrics['terminal_20_step_cm']:.2f} cm")
    axes[1].plot(steps, trace["control_norm"], label="control norm")
    axes[1].plot(steps, trace["action_std"], label="action posterior std")
    axes[1].set_ylabel("control")
    axes[1].legend()
    axes[2].plot(
        steps, trace["observation_std"], label="observation std")
    axes[2].plot(steps, trace["belief_std"], label="belief std")
    axes[2].plot(steps, trace["B_std"], label="q(B) std")
    axes[2].set_ylabel("uncertainty")
    axes[2].set_xlabel("closed-loop step")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output / f"{label}_trace.png", dpi=160)
    plt.close(figure)


def write_evaluation_table(output, rows):
    fields = [
        "phase", "pose", "minimum_cm", "best_20_step_cm",
        "terminal_20_step_cm", "longest_run_below_3cm",
        "terminal_success",
    ]
    with (output / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {name: row[name] for name in fields} for row in rows])
    lines = [
        "| phase | pose | minimum cm | best 20-step cm | "
        "terminal 20-step cm | longest <3 cm | pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['pose']} | "
            f"{row['minimum_cm']:.2f} | "
            f"{row['best_20_step_cm']:.2f} | "
            f"{row['terminal_20_step_cm']:.2f} | "
            f"{row['longest_run_below_3cm']} | "
            f"{'yes' if row['terminal_success'] else 'no'} |")
    (output / "evaluation.md").write_text("\n".join(lines) + "\n")


def save_montage(output, runs, fps):
    import warnings

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    if not runs:
        return
    length = min(len(run["video"]) for run in runs)
    tile_size = runs[0]["video"][0].shape[0] * 3
    montage = []
    for step in range(length):
        canvas = Image.new("RGB", (3 * tile_size, 3 * tile_size), "white")
        for index, run in enumerate(runs[:9]):
            image = Image.fromarray(run["video"][step]).resize(
                (tile_size, tile_size), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, tile_size, 17), fill="white")
            draw.text(
                (2, 2),
                f"{run['label']}  {run['trace']['error_cm'][step]:.2f} cm",
                fill="black")
            canvas.paste(
                image,
                ((index % 3) * tile_size, (index // 3) * tile_size))
        montage.append(np.asarray(canvas))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="os.fork.*")
        imageio.mimsave(
            output / "gate_b_montage.mp4", montage, fps=fps)


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
        require_passed=True)
    reproducibility = manifest.get("reproducibility", {})
    if (
            reproducibility.get("identical") is not True
            or reproducibility.get("validation_runs", 0) < 2):
        raise ManifestError(
            "two reproducible validation passes are required before evaluation")
    if manifest["physical_audit"].get(
            "preserved_failed_seed") != FAILED_AUDIT_SEED:
        raise ManifestError("failed seed 73001 audit record is missing")

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

    def run_pose(label, phase, pose_index, start_pose, goal_pose):
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
        metrics.update({
            "phase": phase,
            "pose": int(pose_index),
            "label": label,
            "seed": (
                SMOKE_SEED if phase == "smoke" else GATE_B_SEED),
        })
        np.savez_compressed(
            artifact["out"] / f"{label}.npz", **trace)
        save_trace_plot(
            artifact["out"], label, trace, metrics, HOLD_THRESHOLD_CM)
        return {
            "label": label,
            "metrics": metrics,
            "trace": trace,
            "video": video,
        }

    runs = []
    report = {
        "protocol": {
            "smoke_seed": SMOKE_SEED,
            "gate_b_seed": int(GATE_B_SEED),
            "threshold_cm": float(HOLD_THRESHOLD_CM),
            "terminal_window": int(HOLD_WINDOW),
            "gate_b_rule": "at least 7 of 8",
            "failed_seed_73001_used_for_tuning": False,
        },
        "smoke": None,
        "gate_b": None,
        "success": False,
    }
    try:
        smoke_start, smoke_goal = sample_task(SMOKE_SEED)
        smoke = run_pose(
            "smoke_73002", "smoke", 0,
            smoke_start, smoke_goal)
        runs.append(smoke)
        report["smoke"] = smoke["metrics"]
        if not smoke["metrics"]["terminal_success"]:
            report["gate_b"] = {
                "run": False,
                "reason": "frozen smoke terminal error was not below 3 cm",
            }
            write_evaluation_table(
                artifact["out"],
                [smoke["metrics"]])
            write_json(
                artifact["out"] / "evaluation.json", report)
            update_runtime(
                artifact["runtime"], "evaluate",
                time.time() - started,
                result="smoke_failed", gate_b_run=False)
            raise RuntimeError(
                "frozen smoke pose failed; Gate B was not run")

        gate_rng = np.random.RandomState(GATE_B_SEED)
        for index in range(8):
            # Derive fixed per-pose seeds once; no outcome-dependent resampling.
            pose_seed = int(gate_rng.randint(0, 2**31 - 1))
            start_pose, goal_pose = sample_task(pose_seed)
            run = run_pose(
                f"gate_b_{index:02d}", "gate_b", index,
                start_pose, goal_pose)
            runs.append(run)
        successes = sum(
            run["metrics"]["terminal_success"]
            for run in runs[1:])
        report["gate_b"] = {
            "run": True,
            "successes": int(successes),
            "poses": 8,
            "passed": bool(successes >= 7),
            "rows": [run["metrics"] for run in runs[1:]],
        }
        report["success"] = bool(successes >= 7)
        rows = [run["metrics"] for run in runs]
        write_evaluation_table(artifact["out"], rows)
        write_json(artifact["out"] / "evaluation.json", report)
        if report["success"] and not args.no_video:
            save_montage(artifact["out"], runs[1:], MONTAGE_FPS)
        update_runtime(
            artifact["runtime"], "evaluate",
            time.time() - started,
            result=("passed" if report["success"] else "gate_b_failed"),
            gate_b_run=True,
            gate_b_successes=int(successes))
        if not report["success"]:
            raise RuntimeError(
                f"Gate B failed honestly: {successes}/8 terminal holds; "
                "no seeds were replaced")
    finally:
        renderer.close()
        environment.close()
    print(
        "Gate B passed: at least 7/8 fresh reachable poses "
        "finished below 3 cm.", flush=True)


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
