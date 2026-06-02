"""Checkpoint validation utilities for JOPA examples.

The validator scores whether a learned observation model and latent dynamics
are plausible enough to use for planning. It intentionally reports simple
numbers: reconstruction error, latent linearity, and one-step latent prediction
error.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .nn.vae import VAE, LOG_STD_CLIP, load_params


@dataclass
class ValidationReport:
    reconstruction_mse: float
    latent_linearity_r2: float
    one_step_latent_mse: float
    n_observations: int
    latent_dim: int
    n_frames: int
    dynamics_source: str
    passed: bool


def _as_vae_batch(observations, n_frames: int):
    obs = jnp.asarray(observations)
    if n_frames == 1:
        if obs.ndim == 4 and obs.shape[1] == 1:
            obs = obs[:, 0]
        if obs.ndim != 3:
            raise ValueError(f"n_frames=1 expects observations shaped (N, 28, 28), got {obs.shape}")
        return obs
    if obs.ndim != 4 or obs.shape[1] != n_frames:
        raise ValueError(f"n_frames={n_frames} expects observations shaped (N, {n_frames}, 28, 28), got {obs.shape}")
    return obs


def encode_observations(model: VAE, params, observations):
    batch = _as_vae_batch(observations, model.n_frames)
    mu, log_std = model.apply(params, batch, method=model.encode)
    return np.asarray(mu), np.asarray(jnp.clip(log_std, *LOG_STD_CLIP))


def reconstruction_mse(model: VAE, params, observations) -> float:
    batch = _as_vae_batch(observations, model.n_frames)
    mu, _ = model.apply(params, batch, method=model.encode)
    recon = model.apply(params, mu, method=model.decode)
    target = batch.reshape(batch.shape[0], -1)
    return float(jnp.mean((recon - target) ** 2))


def fit_linear_dynamics(latents, controls=None):
    latents = np.asarray(latents, dtype=np.float64)
    if len(latents) < 2:
        raise ValueError("at least two latent states are required for one-step validation")
    x, y = latents[:-1], latents[1:]
    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        if len(controls) != len(x):
            raise ValueError(f"controls length {len(controls)} does not match transitions {len(x)}")
        design = np.concatenate([x, controls], axis=1)
    else:
        design = x
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    if controls is None:
        return coef.T, None, pred
    d = latents.shape[1]
    return coef[:d].T, coef[d:].T, pred


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def predict_one_step(latents, A, B=None, controls=None):
    latents = np.asarray(latents, dtype=np.float64)
    pred = latents[:-1] @ np.asarray(A, dtype=np.float64).T
    if B is not None:
        if controls is None:
            raise ValueError("controls are required when B is provided")
        pred = pred + np.asarray(controls, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T
    return pred


def _load_dynamics(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "q_a" not in payload:
        raise ValueError("dynamics checkpoint must be a dict containing at least 'q_a'")
    from .distributions import gaussian_mean

    qa_mean = gaussian_mean(payload["q_a"])
    dim = int(np.sqrt(len(qa_mean)))
    A = np.asarray(qa_mean).reshape(dim, dim)
    B = None
    if payload.get("q_b") is not None:
        qb_mean = gaussian_mean(payload["q_b"])
        B = np.asarray(qb_mean).reshape(dim, -1)
    return A, B


def validate_checkpoint(
    model: VAE,
    params,
    observations,
    *,
    controls=None,
    dynamics=None,
    max_reconstruction_mse: float | None = None,
    min_latent_linearity_r2: float | None = None,
    max_one_step_latent_mse: float | None = None,
) -> ValidationReport:
    latents, _ = encode_observations(model, params, observations)
    A_fit, B_fit, fit_pred = fit_linear_dynamics(latents, controls)
    linearity_r2 = r2_score(latents[1:], fit_pred)

    if dynamics is None:
        A, B = A_fit, B_fit
        source = "fitted_latents"
    else:
        A, B = dynamics
        source = "checkpoint"
    pred = predict_one_step(latents, A, B, controls if B is not None else None)
    one_step_mse = float(np.mean((latents[1:] - pred) ** 2))
    recon_mse = reconstruction_mse(model, params, observations)

    passed = True
    if max_reconstruction_mse is not None:
        passed = passed and recon_mse <= max_reconstruction_mse
    if min_latent_linearity_r2 is not None:
        passed = passed and not np.isnan(linearity_r2) and linearity_r2 >= min_latent_linearity_r2
    if max_one_step_latent_mse is not None:
        passed = passed and one_step_mse <= max_one_step_latent_mse

    return ValidationReport(
        reconstruction_mse=recon_mse,
        latent_linearity_r2=float(linearity_r2),
        one_step_latent_mse=one_step_mse,
        n_observations=int(len(latents)),
        latent_dim=int(model.latent_dim),
        n_frames=int(model.n_frames),
        dynamics_source=source,
        passed=bool(passed),
    )


def _load_optional_controls(path: Path | None):
    if path is None:
        return None
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a JOPA VAE/dynamics checkpoint before planning.")
    parser.add_argument("--vae", type=Path, required=True, help="Path to VAE params saved by jopa.nn.vae.save_params.")
    parser.add_argument("--sequence", type=Path, required=True, help="Numpy .npy observation sequence.")
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--n-frames", type=int, default=1)
    parser.add_argument("--dynamics", type=Path, help="Optional pickle with q_a and optional q_b, as saved by examples.")
    parser.add_argument("--controls", type=Path, help="Optional .npy controls, length N-1.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--max-reconstruction-mse", type=float)
    parser.add_argument("--min-latent-linearity-r2", type=float)
    parser.add_argument("--max-one-step-latent-mse", type=float)
    args = parser.parse_args(argv)

    model = VAE(latent_dim=args.latent_dim, n_frames=args.n_frames)
    params = load_params(model, args.vae)
    observations = np.load(args.sequence)
    controls = _load_optional_controls(args.controls)
    dynamics = _load_dynamics(args.dynamics) if args.dynamics is not None else None
    report = validate_checkpoint(
        model,
        params,
        observations,
        controls=controls,
        dynamics=dynamics,
        max_reconstruction_mse=args.max_reconstruction_mse,
        min_latent_linearity_r2=args.min_latent_linearity_r2,
        max_one_step_latent_mse=args.max_one_step_latent_mse,
    )
    payload = asdict(report)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
