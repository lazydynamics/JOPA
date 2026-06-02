"""Diagnostics for belief updates and learned linear-Gaussian transitions."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .blocks import JointModel
from .distributions import Gaussian, combine_gaussians, gaussian_mean_cov, wishart_mean


@dataclass
class BeliefGap:
    """Difference between two Gaussian beliefs."""

    kl: float
    mahalanobis: float
    mean_l2: float
    trace_before: float
    trace_after: float
    trace_delta: float


@dataclass
class TransitionMetrics:
    """One-step transition residual metrics."""

    n: int
    dim: int
    r2: float
    mae: float
    rmse: float
    mean_nis: float
    mean_nis_per_dim: float
    mean_nll: float


@dataclass
class FilterStepDiagnostics:
    """Per-step diagnostics emitted by `filter_with_diagnostics`."""

    step: int
    block: str
    predictive_corrected: BeliefGap | None
    corrected_fused: BeliefGap | None
    observation_trace: float
    corrected_trace: float
    fused_trace: float


def _mean_cov(g: Gaussian):
    mean, cov = gaussian_mean_cov(g)
    return np.asarray(mean, dtype=np.float64), np.asarray(cov, dtype=np.float64)


def covariance_trace(g: Gaussian) -> float:
    """Trace of the covariance represented by an information-form Gaussian."""
    _, cov = _mean_cov(g)
    return float(np.trace(cov))


def gaussian_kl(q: Gaussian, p: Gaussian) -> float:
    """Compute KL(q || p) for full-covariance Gaussians."""
    mq, cq = _mean_cov(q)
    mp, cp = _mean_cov(p)
    dim = mq.shape[0]
    precision_p = np.linalg.inv(cp)
    diff = mp - mq
    sign_q, logdet_q = np.linalg.slogdet(cq)
    sign_p, logdet_p = np.linalg.slogdet(cp)
    if sign_q <= 0 or sign_p <= 0:
        return float("nan")
    return float(
        0.5
        * (
            np.trace(precision_p @ cq)
            + diff @ precision_p @ diff
            - dim
            + logdet_p
            - logdet_q
        )
    )


def belief_gap(before: Gaussian, after: Gaussian) -> BeliefGap:
    """Summarize a belief update from `before` to `after`.

    This is the cheap surprise signal discussed in the project roadmap:
    predictive belief -> corrected/fused belief, with precision/covariance
    changes exposed explicitly.
    """
    mb, cb = _mean_cov(before)
    ma, _ = _mean_cov(after)
    delta = ma - mb
    precision_before = np.linalg.inv(cb)
    trace_before = covariance_trace(before)
    trace_after = covariance_trace(after)
    return BeliefGap(
        kl=gaussian_kl(after, before),
        mahalanobis=float(delta @ precision_before @ delta),
        mean_l2=float(np.linalg.norm(delta)),
        trace_before=trace_before,
        trace_after=trace_after,
        trace_delta=float(trace_after - trace_before),
    )


def transition_residuals(A, B, states, actions=None):
    """Return `(truth, pred, residual)` for a linear transition."""
    states = np.asarray(states, dtype=np.float64)
    truth = states[1:]
    pred = states[:-1] @ np.asarray(A, dtype=np.float64).T
    if B is not None and actions is not None:
        pred = pred + np.asarray(actions, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T
    n = min(len(truth), len(pred))
    truth, pred = truth[:n], pred[:n]
    return truth, pred, truth - pred


def transition_metrics(residuals, truth, precision) -> TransitionMetrics:
    """Compute R2, residual magnitudes, NIS, and Gaussian NLL."""
    residuals = np.asarray(residuals, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    n, dim = residuals.shape
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    nis = np.einsum("bi,ij,bj->b", residuals, precision, residuals)
    sign, logdet = np.linalg.slogdet(precision)
    mean_nll = float("nan")
    if sign > 0:
        mean_nll = float(np.mean(0.5 * (dim * np.log(2.0 * np.pi) - logdet + nis)))
    return TransitionMetrics(
        n=int(n),
        dim=int(dim),
        r2=float(1.0 - ss_res / max(ss_tot, 1e-12)),
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mean_nis=float(np.mean(nis)),
        mean_nis_per_dim=float(np.mean(nis) / max(dim, 1)),
        mean_nll=mean_nll,
    )


def block_precision(block) -> jnp.ndarray:
    """Expected transition precision for a learned block."""
    return wishart_mean(block.transition.q_W)


def filter_with_diagnostics(model: JointModel, stream) -> tuple[dict[str, Gaussian], list[FilterStepDiagnostics]]:
    """Run `JointModel.filter` while recording belief-update diagnostics.

    The filtering semantics match `JointModel.filter`: each block receives an
    observation message, learned/known transitions produce predictive beliefs
    after the first step, and configured couplings fuse beliefs after each step.
    """
    beliefs = {b.name: None for b in model.blocks}
    rows: list[FilterStepDiagnostics] = []
    for step_idx, step in enumerate(stream):
        u = step.get("control")
        corrected = {}
        for b in model.blocks:
            obs = b.observation.message(step[b.name])
            predicted = None
            if beliefs[b.name] is None:
                corrected[b.name] = obs
            else:
                predicted = b.transition.predict(beliefs[b.name], u)
                corrected[b.name] = combine_gaussians(predicted, obs)
            rows.append(
                FilterStepDiagnostics(
                    step=step_idx,
                    block=b.name,
                    predictive_corrected=belief_gap(predicted, corrected[b.name]) if predicted is not None else None,
                    corrected_fused=None,
                    observation_trace=covariance_trace(obs),
                    corrected_trace=covariance_trace(corrected[b.name]),
                    fused_trace=covariance_trace(corrected[b.name]),
                )
            )
        beliefs = corrected
        for c in model.couplings:
            before_from, before_to = beliefs[c.from_name], beliefs[c.to_name]
            beliefs[c.from_name], beliefs[c.to_name] = c.fuse(before_from, before_to)
            for name, before in ((c.from_name, before_from), (c.to_name, before_to)):
                rows.append(
                    FilterStepDiagnostics(
                        step=step_idx,
                        block=name,
                        predictive_corrected=None,
                        corrected_fused=belief_gap(before, beliefs[name]),
                        observation_trace=float("nan"),
                        corrected_trace=covariance_trace(before),
                        fused_trace=covariance_trace(beliefs[name]),
                    )
                )
    return beliefs, rows
