"""State-free validation utilities for the pixel-only Reacher example.

These checks deliberately accept encoded observations and motor commands only.
They are kept separate from MuJoCo evaluation so representation/model tuning
cannot accidentally depend on privileged simulator coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def rollout_diagnostics(transition, encoded, *, horizon=6, samples=None,
                        seed=0, localize=True):
    """Validate frozen-horizon latent prediction on held-out trajectories.

    A local posterior is selected at the first encoded observation and then
    held fixed for the complete rollout, matching the local-linear model used
    by one planning call.  Normalization is computed from held-out latents.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    candidates = [
        (sequence_index, start)
        for sequence_index, (states, controls) in enumerate(encoded)
        for start in range(len(controls) - horizon + 1)
    ]
    if not candidates:
        raise ValueError("held-out data contain no complete rollout")
    if samples is None:
        chosen = np.arange(len(candidates))
    else:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(
            len(candidates), min(samples, len(candidates)), replace=False)
    feature_mean = np.concatenate([x for x, _ in encoded]).mean(0)

    one_num = one_den = rollout_num = rollout_den = 0.0
    entropies = []
    for item in chosen:
        sequence_index, start = candidates[item]
        states, controls = encoded[sequence_index]
        state = np.asarray(states[start]).copy()
        if localize and hasattr(transition, "localize"):
            transition.localize(state)
            weights = getattr(transition, "last_expert_weights", None)
            if weights is not None:
                weights = np.asarray(weights)
                entropies.append(float(-np.sum(
                    weights * np.log(np.maximum(weights, 1e-12)))))
        A = np.asarray(transition.A)
        B = np.asarray(transition.B)
        c = np.asarray(transition.c)
        one = A @ state + B @ controls[start] + c
        one_target = states[start + 1]
        one_num += float(np.sum((one - one_target) ** 2))
        one_den += float(np.sum((one_target - feature_mean) ** 2))
        for step in range(horizon):
            state = A @ state + B @ controls[start + step] + c
        target = states[start + horizon]
        rollout_num += float(np.sum((state - target) ** 2))
        rollout_den += float(np.sum((target - feature_mean) ** 2))

    B = np.asarray(transition.B)
    B_std = np.asarray(transition.B_std)
    return {
        "one_step_nrmse": float(np.sqrt(one_num / max(one_den, 1e-12))),
        f"{horizon}_step_nrmse": float(np.sqrt(
            rollout_num / max(rollout_den, 1e-12))),
        "mean_B_signal_to_std": float(np.mean(
            np.abs(B) / np.maximum(B_std, 1e-8))),
        "mean_expert_entropy": (
            float(np.mean(entropies)) if entropies else None),
        "samples": len(chosen),
    }


def dynamics_gate(report, *, horizon=6, one_step_max=0.35,
                  rollout_max=0.50, control_signal_min=1.0):
    """Predeclared state-free admission gate for closed-loop evaluation."""
    checks = {
        "one_step": report["one_step_nrmse"] < one_step_max,
        "rollout": report[f"{horizon}_step_nrmse"] < rollout_max,
        "control_identified": (
            report["mean_B_signal_to_std"] > control_signal_min),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}



ADMISSION_THRESHOLDS = {
    "one_step_nrmse_max": 0.35,
    "six_step_nrmse_max": 0.50,
    "control_signal_to_std_min": 1.0,
    "pose_foreground_spearman_min": 0.75,
    "repeated_motion_rms_max": 0.10,
    "zero_action_drift_20_max": 0.15,
    "innovation_coverage_min": 0.85,
    "innovation_coverage_max": 0.95,
}
MANIFEST_SCHEMA = "jopa-pose-motion-reacher-v1"
FAILED_AUDIT_SEED = 73001
SMOKE_SEED = 73002


class ManifestError(RuntimeError):
    """A manifest is missing, incompatible, stale, or failed."""


def split_trajectories(
        trajectories, seed=41, ratios=(0.8, 0.1, 0.1)):
    """Return deterministic, trajectory-disjoint train/calibration/test data."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    count = len(trajectories)
    if count < 3:
        raise ValueError("at least three trajectories are required")
    order = np.random.RandomState(seed).permutation(count)
    calibration_count = max(1, int(round(ratios[1] * count)))
    test_count = max(1, int(round(ratios[2] * count)))
    if calibration_count + test_count >= count:
        calibration_count = test_count = 1
    train_count = count - calibration_count - test_count
    indices = {
        "train": order[:train_count].astype(int).tolist(),
        "calibration": order[
            train_count:train_count + calibration_count].astype(int).tolist(),
        "test": order[train_count + calibration_count:].astype(int).tolist(),
    }
    sets = {
        name: [trajectories[index] for index in selected]
        for name, selected in indices.items()
    }
    return {
        **sets,
        "indices": indices,
        "seed": int(seed),
        "ratios": [float(value) for value in ratios],
    }


trajectory_split = split_trajectories


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
                end < len(values)
                and values[order[end]] == values[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(first, second):
    """Dependency-free Spearman correlation with average tie ranks."""
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or len(first) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_rank, second_rank) / denominator)


def foreground_mismatch(first, second, background=None):
    """Foreground-weighted RMS mismatch for paired grayscale frames."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("paired images must have equal shape")
    if background is None:
        background = np.median(
            np.concatenate([
                first.reshape((-1,) + first.shape[-2:]),
                second.reshape((-1,) + second.shape[-2:]),
            ]), axis=0)
    background = np.asarray(background, dtype=np.float64)
    if background.shape != first.shape[-2:]:
        raise ValueError("background must match image height and width")
    foreground = np.maximum(
        np.abs(first - background), np.abs(second - background))
    numerator = np.sum(
        foreground * (first - second) ** 2, axis=(-2, -1))
    denominator = np.sum(foreground, axis=(-2, -1))
    fallback = np.mean((first - second) ** 2, axis=(-2, -1))
    mean_square = np.where(
        denominator > 1e-12,
        numerator / np.maximum(denominator, 1e-12),
        fallback)
    return np.sqrt(np.maximum(mean_square, 0.0))


def visual_aliasing_metrics(
        images, states, *, pose_dim=4, background=None,
        pairs=4096, seed=0):
    """Distance preservation and latent-neighbour visual-aliasing audit."""
    images = np.asarray(images, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    if images.ndim != 3 or len(images) != len(states):
        raise ValueError("images (N,H,W) and states (N,D) must align")
    if states.ndim != 2 or not 0 < pose_dim <= states.shape[1]:
        raise ValueError("pose_dim must index the state columns")
    if len(images) < 3:
        raise ValueError("visual metrics require at least three images")
    if background is None:
        background = np.median(images, axis=0)
    scale = np.maximum(states.std(axis=0), 1e-6)
    standardized = (states - states.mean(axis=0)) / scale
    pose = standardized[:, :pose_dim]

    rng = np.random.RandomState(seed)
    pair_count = min(int(pairs), len(images) * (len(images) - 1))
    first = rng.randint(0, len(images), size=pair_count)
    second = rng.randint(0, len(images) - 1, size=pair_count)
    second += second >= first
    pose_distance = np.linalg.norm(
        pose[first] - pose[second], axis=1) / np.sqrt(pose_dim)
    visual_distance = foreground_mismatch(
        images[first], images[second], background)
    correlation = spearman_correlation(
        pose_distance, visual_distance)

    squared = np.sum(
        (standardized[:, None] - standardized[None, :]) ** 2,
        axis=2)
    np.fill_diagonal(squared, np.inf)
    nearest = np.argmin(squared, axis=1)
    nearest_mismatch = foreground_mismatch(
        images, images[nearest], background)

    random_second = rng.randint(0, len(images) - 1, size=len(images))
    random_second += random_second >= np.arange(len(images))
    random_mismatch = foreground_mismatch(
        images, images[random_second], background)
    nearest_p90 = float(np.quantile(nearest_mismatch, 0.90))
    random_p20 = float(np.quantile(random_mismatch, 0.20))
    return {
        "pose_foreground_spearman": float(correlation),
        "nearest_neighbor_visual_mismatch_p90": nearest_p90,
        "random_pair_visual_mismatch_p20": random_p20,
        "nearest_below_random": bool(nearest_p90 < random_p20),
        "visual_pairs": int(pair_count),
    }


def repeated_frame_motion_rms(
        repeated_states, reference_states=None, *, pose_dim=4):
    """RMS rest motion after standardization by replay latent scales."""
    repeated = np.asarray(repeated_states, dtype=np.float64)
    if repeated.ndim != 2 or repeated.shape[1] <= pose_dim:
        raise ValueError("repeated states must have pose and motion columns")
    reference = (
        repeated if reference_states is None
        else np.asarray(reference_states, dtype=np.float64))
    if reference.ndim != 2 or reference.shape[1] != repeated.shape[1]:
        raise ValueError("reference states must match latent width")
    motion_scale = np.maximum(
        reference[:, pose_dim:].std(axis=0), 1e-6)
    return float(np.sqrt(np.mean(
        (repeated[:, pose_dim:] / motion_scale) ** 2)))


def zero_action_fixed_point_drift(
        transition, rest_states, reference_states=None, *, steps=20):
    """Zero-action drift of static pixel states, normalized by test latents."""
    if steps < 1:
        raise ValueError("steps must be positive")
    start = np.asarray(rest_states, dtype=np.float64)
    if start.ndim != 2:
        raise ValueError("rest_states must have shape (N,D)")
    reference = (
        start if reference_states is None
        else np.asarray(reference_states, dtype=np.float64))
    if reference.ndim != 2 or reference.shape[1] != start.shape[1]:
        raise ValueError("reference_states must match rest state width")
    scale = np.maximum(reference.std(axis=0), 1e-6)
    state = start.copy()
    matrix = np.asarray(transition.A, dtype=np.float64)
    drift = getattr(transition, "c", None)
    offset = (
        np.zeros(start.shape[1], dtype=np.float64)
        if drift is None else np.asarray(drift, dtype=np.float64))
    for _ in range(steps):
        state = state @ matrix.T + offset
    return float(np.sqrt(np.mean(((state - start) / scale) ** 2)))


def pixel_action_innovations(transition, encoded):
    """One-step innovations and target log stds from encoded replay."""
    innovations = []
    log_stds = []
    matrix = np.asarray(transition.A)
    control_matrix = np.asarray(transition.B)
    drift = getattr(transition, "c", None)
    offset = (
        np.zeros(matrix.shape[0])
        if drift is None else np.asarray(drift))
    for item in encoded:
        if isinstance(item, dict):
            states = np.asarray(item["means"])
            logs = np.asarray(item["log_stds"])
            controls = np.asarray(item["controls"])
        elif len(item) == 3:
            states, logs, controls = map(np.asarray, item)
        else:
            raise ValueError(
                "calibration replay needs means, log_stds, and controls")
        if controls.shape[0] != states.shape[0] - 1:
            raise ValueError("one control is required per latent transition")
        prediction = (
            states[:-1] @ matrix.T
            + controls @ control_matrix.T + offset)
        innovations.append(states[1:] - prediction)
        log_stds.append(logs[1:])
    if not innovations:
        raise ValueError("at least one encoded trajectory is required")
    return np.concatenate(innovations), np.concatenate(log_stds)


def innovation_interval_coverage(
        innovations, *, log_stds=None, standard_deviations=None,
        level=0.90):
    """Empirical elementwise Gaussian innovation-interval coverage."""
    from statistics import NormalDist

    innovation = np.asarray(innovations, dtype=np.float64)
    if (log_stds is None) == (standard_deviations is None):
        raise ValueError("provide exactly one uncertainty representation")
    standard_deviation = (
        np.exp(np.asarray(log_stds, dtype=np.float64))
        if log_stds is not None
        else np.asarray(standard_deviations, dtype=np.float64))
    if innovation.shape != standard_deviation.shape:
        raise ValueError("innovations and uncertainties must have equal shape")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    inside = np.abs(innovation) <= (
        z_value * np.maximum(standard_deviation, 1e-12))
    return float(np.mean(inside))


def admission_gate(report, *, thresholds=None):
    """Apply every predeclared state-free gate without short-circuiting."""
    limits = dict(ADMISSION_THRESHOLDS)
    if thresholds is not None:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(
                f"unknown admission thresholds: {sorted(unknown)}")
        limits.update(thresholds)

    def finite_compare(key, operation):
        value = report.get(key, np.nan)
        return bool(np.isfinite(value) and operation(float(value)))

    checks = {
        "one_step_latent_nrmse": finite_compare(
            "one_step_nrmse",
            lambda value: value < limits["one_step_nrmse_max"]),
        "six_step_latent_nrmse": finite_compare(
            "6_step_nrmse",
            lambda value: value < limits["six_step_nrmse_max"]),
        "control_identified": finite_compare(
            "mean_B_signal_to_std",
            lambda value: value > limits["control_signal_to_std_min"]),
        "pose_visual_distance": finite_compare(
            "pose_foreground_spearman",
            lambda value: value >= limits[
                "pose_foreground_spearman_min"]),
        "visual_neighbour_aliasing": bool(
            np.isfinite(report.get(
                "nearest_neighbor_visual_mismatch_p90", np.nan))
            and np.isfinite(report.get(
                "random_pair_visual_mismatch_p20", np.nan))
            and report["nearest_neighbor_visual_mismatch_p90"]
            < report["random_pair_visual_mismatch_p20"]),
        "repeated_frame_motion": finite_compare(
            "repeated_frame_motion_rms",
            lambda value: value < limits["repeated_motion_rms_max"]),
        "zero_action_fixed_point": finite_compare(
            "zero_action_drift_20_nrmse",
            lambda value: value < limits[
                "zero_action_drift_20_max"]),
        "innovation_calibration": finite_compare(
            "innovation_90_coverage",
            lambda value: (
                limits["innovation_coverage_min"]
                <= value
                <= limits["innovation_coverage_max"])),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": limits,
    }


def checkpoint_sha256(path, chunk_size=1024 * 1024):
    import hashlib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path, manifest):
    """Atomically persist a JSON manifest."""
    import json
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(path):
    import json

    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"could not read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError("manifest root must be a JSON object")
    return value


def validate_manifest(
        manifest_or_path, *, expected_architecture=None,
        checkpoint_paths=None, require_passed=True):
    """Refuse stale architecture, hashes, splits, or failed gate results."""
    manifest = (
        load_manifest(manifest_or_path)
        if not isinstance(manifest_or_path, dict)
        else manifest_or_path)
    errors = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("unsupported or missing schema")
    split = manifest.get("split", {})
    if not isinstance(split.get("seed"), int):
        errors.append("missing split seed")
    indices = split.get("indices", {})
    if set(indices) != {"train", "calibration", "test"}:
        errors.append("missing trajectory split indices")
    else:
        flattened = [
            item for name in ("train", "calibration", "test")
            for item in indices[name]]
        if len(flattened) != len(set(flattened)):
            errors.append("trajectory splits overlap")
    architecture = manifest.get("architecture")
    if expected_architecture is not None:
        if architecture != expected_architecture:
            errors.append("sensor architecture is incompatible")
    elif not isinstance(architecture, dict):
        errors.append("missing sensor architecture")

    checkpoints = manifest.get("checkpoints", {})
    paths = {} if checkpoint_paths is None else checkpoint_paths
    for name, record in checkpoints.items():
        if not isinstance(record, dict) or "sha256" not in record:
            errors.append(f"checkpoint {name} has no hash")
            continue
        candidate = paths.get(name, record.get("path"))
        if candidate is None:
            errors.append(f"checkpoint {name} has no path")
            continue
        try:
            actual = checkpoint_sha256(candidate)
        except OSError as error:
            errors.append(f"checkpoint {name} unavailable: {error}")
            continue
        if actual != record["sha256"]:
            errors.append(f"checkpoint {name} hash mismatch")
    for required in ("sensor", "dynamics"):
        if required not in checkpoints:
            errors.append(f"missing {required} checkpoint")

    gates = manifest.get("admission")
    if require_passed and not isinstance(gates, dict):
        errors.append("missing admission gates")
    elif isinstance(gates, dict):
        expected_checks = set(admission_gate({})["checks"])
        if set(gates.get("checks", {})) != expected_checks:
            errors.append("manifest does not contain every admission gate")
        if require_passed and gates.get("passed") is not True:
            errors.append("state-free admission gates did not pass")
    if errors:
        raise ManifestError("; ".join(errors))
    return manifest
