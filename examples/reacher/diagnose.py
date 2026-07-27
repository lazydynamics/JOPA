"""Pixel-only diagnostics: latent rollout, aliasing, calibration coverage."""
from __future__ import annotations

import copy
import json
import pickle
import time

import numpy as np

from jopa import PoseMotionObservation
from jopa.nn.vae import load_params

from .artifacts import (
    VALIDATION_REPEATS,
    attach_encoded_replay,
    architecture_for,
    checkpoint_record,
    encodings_for,
    fit_conjugate_dynamics,
    paths,
    sensor_for,
    split_replay,
    update_runtime,
    write_json,
)
from .runtime import HORIZON, MOTION_DIM, NEIGHBORS, POSE_DIM, REFIT_OBS_PRECISION
from .validation import (
    checkpoint_sha256,
    innovation_interval_coverage,
    load_manifest,
    pixel_action_innovations,
    repeated_frame_motion_rms,
    rollout_diagnostics,
    validate_manifest,
    visual_aliasing_metrics,
    write_manifest,
    zero_action_fixed_point_drift,
)


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
    return report


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


