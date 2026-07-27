"""Sensor training and the bootstrap conjugate fit."""
from __future__ import annotations

import json
import pickle
import time

import jax
import numpy as np

from jopa import PoseMotionObservation
from jopa.nn.vae import load_params, save_params, train_pose_motion_vae

from .artifacts import (
    PREDICTION_WEIGHT,
    SENSOR_SEED,
    ManifestError,
    base_manifest,
    cache_path,
    checkpoint_record,
    checkpoint_sha256,
    encodings_for,
    fit_conjugate_dynamics,
    json_value,
    load_manifest,
    paths,
    sensor_for,
    split_replay,
    update_runtime,
    write_json,
    write_manifest,
)
from .runtime import (
    ACTION_PRECISION_HOLD,
    ACTION_PRECISION_TRAVEL,
    ACTION_PRIOR_POWER,
    ACTION_PRIOR_SCALE,
    IMG_SIZE,
    MOTION_DIM,
    MOTION_GOAL_PRECISION,
    MOTION_HIDDEN_DIM,
    N_FRAMES,
    POSE_DIM,
    POSE_GOAL_PRECISION,
    SUBGOAL_TRUST,
)
from .spec import NEIGHBORS, REFIT_OBS_PRECISION


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
    manifest = base_manifest(args, split, provenance, policy={
        "pose_goal_precision": float(POSE_GOAL_PRECISION),
        "motion_goal_precision": float(MOTION_GOAL_PRECISION),
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
    })
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


