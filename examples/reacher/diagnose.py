"""Pixel-only diagnostics: latent rollout error and innovation calibration."""
from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np

from jopa import PoseMotionObservation
from jopa.nn.vae import load_params

from .artifacts import (
    architecture_for,
    attach_encoded_replay,
    checkpoint_record,
    checkpoint_sha256,
    encodings_for,
    fit_conjugate_dynamics,
    load_manifest,
    paths,
    sensor_for,
    split_replay,
    update_runtime,
    validate_manifest,
    write_json,
    write_manifest,
)
from .metrics import (
    innovation_interval_coverage,
    pixel_action_innovations,
    rollout_diagnostics,
)
from .runtime import HORIZON, MOTION_DIM, NEIGHBORS, POSE_DIM, REFIT_OBS_PRECISION


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


def validation_report(
        transition, observation, encoded_test,
        replay_encoded=None, neighbors=NEIGHBORS,
        refit_obs_prec=REFIT_OBS_PRECISION):
    """Latent prediction error and innovation calibration on held-out replay.

    The deployed model is the conjugate posterior refit locally at the belief,
    so the rollout is measured on that; the innovation check uses the global
    posterior, which is what the filter starts each episode from.
    """
    if replay_encoded is not None:
        local = attach_encoded_replay(
            copy.deepcopy(transition), replay_encoded, neighbors,
            obs_prec=refit_obs_prec)
    else:
        local = transition
    rollout = rollout_diagnostics(
        local, encoded_tuples(encoded_test),
        horizon=HORIZON, samples=None, localize=replay_encoded is not None)
    innovations, base_log_stds = pixel_action_innovations(
        transition, encoded_test)
    calibrated_logs = np.clip(
        base_log_stds + np.asarray(observation.log_std_offsets), -6.0, 2.0)
    coverage = innovation_interval_coverage(
        innovations, log_stds=calibrated_logs, level=0.90)
    return {
        **rollout,
        "innovation_90_coverage": float(coverage),
        "test_trajectories": len(encoded_test),
    }


def run_validate(args):
    started = time.time()
    artifact = paths(args)
    manifest = load_manifest(artifact["manifest"])
    validate_manifest(
        manifest,
        expected_architecture=architecture_for(args),
        checkpoint_paths={
            "sensor": artifact["sensor"],
            "background": artifact["background"],
        },
    )
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
    report = validation_report(
        transition, observation, encoded["test"],
        replay_encoded=encoded["train"])

    manifest["checkpoints"]["dynamics"] = checkpoint_record(
        artifact["dynamics"])
    manifest["calibration"] = {
        "source_split": "calibration",
        "method": "per-dimension 90% innovation quantile matching",
        "log_std_offsets": offsets.tolist(),
        "target_coverage": 0.90,
    }
    manifest["diagnostics"] = {"source_split": "test", **report}
    manifest["phase"] = "validated"
    write_manifest(artifact["manifest"], manifest)
    write_json(artifact["out"] / "diagnostics.json", report)
    update_runtime(artifact["runtime"], "validate", time.time() - started)
    print(json.dumps(report, indent=2), flush=True)
