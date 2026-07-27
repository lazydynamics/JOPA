"""Pixel-only metrics: latent rollout error and innovation calibration.

These accept encoded observations and motor commands only, never simulator
coordinates, so sensor and model tuning cannot depend on privileged state.
"""
from __future__ import annotations

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


