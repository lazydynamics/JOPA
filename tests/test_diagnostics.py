import numpy as np
import jax.numpy as jnp

from jopa import (
    Block, Gaussian, JointModel, KnownPhysics,
    belief_gap, covariance_trace, filter_with_diagnostics, gaussian_kl,
)


def _msg(mean, prec):
    mean = jnp.asarray(mean, dtype=float)
    lam = prec * jnp.eye(mean.shape[0])
    return Gaussian(eta=lam @ mean, lam=lam)


def test_gaussian_kl_is_zero_for_identical_beliefs():
    g = _msg([1.0, -2.0], 3.0)
    assert gaussian_kl(g, g) < 1e-8


def test_belief_gap_reports_precision_tightening():
    before = _msg([0.0, 0.0], 1.0)
    after = _msg([0.2, 0.0], 4.0)
    gap = belief_gap(before, after)
    assert gap.kl > 0.0
    assert gap.mahalanobis > 0.0
    assert gap.trace_after < gap.trace_before
    assert gap.trace_delta < 0.0


def test_filter_with_diagnostics_matches_filter_final_belief():
    identity = lambda m: (jnp.eye(2), None, jnp.zeros(2))
    block = Block("z", KnownPhysics(2, linearize=identity), observe=lambda d: _msg(d, 1e3))
    stream = [{"z": np.array([0.1, 0.0])}, {"z": np.array([0.2, 0.0])}]
    model = JointModel([block])
    plain = model.filter(stream)
    traced, rows = filter_with_diagnostics(model, stream)
    assert rows[0].predictive_corrected is None
    assert rows[1].predictive_corrected is not None
    assert covariance_trace(traced["z"]) == covariance_trace(plain["z"])
