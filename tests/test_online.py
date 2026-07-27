"""Semantic tests for continual learning: affine drift (offset) and
posterior-carryover with exponential forgetting (remember)."""
import jax.numpy as jnp
import numpy as np
import pytest

from jopa.blocks import Block, JointModel, LearnedLinear
from jopa.distributions import (
    Gaussian,
    gaussian_logpdf,
    gaussian_mean,
    gaussian_mean_cov,
)


def _msg(mean, prec):
    mean = jnp.asarray(mean, dtype=float)
    lam = prec * jnp.eye(mean.shape[0])
    return Gaussian(eta=lam @ mean, lam=lam)


def _drift_traj(A, c, T, obs_std, proc_std, rng, B=None, us=None, init_scale=1.0):
    """Short trajectory from a random init — drift is identified against A by
    transients from diverse starting points (a steady state hides it)."""
    d = A.shape[0]
    x = init_scale * rng.randn(d)
    seq = []
    for t in range(T):
        seq.append(_msg(x + obs_std * rng.randn(d), 1.0 / obs_std ** 2))
        x = A @ x + c + proc_std * rng.randn(d)
        if B is not None:
            x = x + B @ us[t]
    return seq


def _drift_seqs(A, c, rng, n=30, T=10, **kw):
    return [_drift_traj(A, c, T, 0.02, 0.01, rng, **kw) for _ in range(n)]


# ============================================================================
# gaussian_logpdf
# ============================================================================

def test_gaussian_logpdf_matches_closed_form():
    mu, var = jnp.array([1.0]), 0.25
    g = Gaussian(eta=jnp.array([mu[0] / var]), lam=jnp.array([[1.0 / var]]))
    x = jnp.array([1.5])
    expected = -0.5 * ((0.5 ** 2) / var + np.log(2 * np.pi * var))
    assert np.allclose(float(gaussian_logpdf(g, x)), expected, atol=1e-6)


# ============================================================================
# LearnedLinear(offset=True) — learned constant drift via the pinned-1 channel
# ============================================================================

def test_offset_recovers_uncontrolled_drift():
    """x' = A·x + c: with offset=True, `c` lands on the true drift and `A` is
    not contaminated by it."""
    rng = np.random.RandomState(0)
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    c = np.array([0.20, -0.10])
    tr = LearnedLinear(dim=2, offset=True, n_iterations=40)
    tr.learn(_drift_seqs(A, c, rng))
    assert tr.B is None                       # du=0 — no force columns
    assert np.allclose(np.array(tr.c), c, atol=0.04)
    assert np.allclose(np.array(tr.A), A, atol=0.04)


def test_offset_separates_drift_from_control():
    rng = np.random.RandomState(1)
    A = np.array([[0.6, 0.0], [0.0, 0.6]])
    B = np.array([[0.2], [0.1]])
    c = np.array([-0.20, 0.30])
    seqs, ctrls = [], []
    for _ in range(30):
        us = [rng.randn(1) for _ in range(9)] + [np.zeros(1)]
        seqs.append(_drift_traj(A, c, 10, 0.02, 0.01, rng, B=B, us=us))
        ctrls.append([jnp.asarray(u) for u in us[:9]])
    tr = LearnedLinear(dim=2, du=1, offset=True, n_iterations=40)
    tr.learn(seqs, ctrls)
    assert np.allclose(np.array(tr.B), B, atol=0.04)
    assert np.allclose(np.array(tr.c), c, atol=0.04)


def test_offset_predict_and_smooth_apply_drift_without_controls():
    """`predict` and the `smooth` rollout advance the belief by A·x + c even
    when no controls are supplied."""
    rng = np.random.RandomState(2)
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    c = np.array([0.20, -0.10])
    block = Block("z", LearnedLinear(dim=2, offset=True, n_iterations=40),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    block.transition.learn(_drift_seqs(A, c, rng))
    A_hat, c_hat = np.array(block.transition.A), np.array(block.transition.c)
    # predict: belief advances by the learned A·x + c with u=None
    nxt = block.transition.predict(_msg(jnp.array([1.0, 1.0]), 1e4))
    assert np.allclose(np.array(gaussian_mean(nxt)), A_hat @ np.ones(2) + c_hat, atol=0.02)
    # smooth: the n_predict rollout also carries the drift
    raws = [np.array([0.5, 0.5]) * (0.9 ** t) for t in range(30)]
    out = model.smooth({"z": raws}, n_predict=1)["z"]
    m = np.array(out["means"])
    assert np.allclose(m[30], A_hat @ m[29] + c_hat, atol=0.02)


def test_plan_with_offset_compensates_drift():
    """Under a constant drift, planned actions counteract it: the plan reaches
    the goal when rolled out under the true drifting dynamics."""
    rng = np.random.RandomState(3)
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    B = np.array([[0.0], [0.1]])
    c = np.array([0.0, 0.05])
    trajs = []
    for _ in range(15):
        x = rng.randn(2) * 0.5
        seq, us = [x.copy()], []
        for _ in range(39):
            u = rng.randn(1)
            us.append(u)
            x = A @ x + (B @ u).ravel() + c + 0.005 * rng.randn(2)
            seq.append(x.copy())
        trajs.append({"z": seq, "control": us})
    block = Block("z", LearnedLinear(dim=2, du=1, offset=True, n_iterations=40),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    model.learn(trajs)
    actions = model.plan({"z": [np.array([0.0, 0.0])] + [None] * 18 + [np.array([1.0, 0.0])]},
                         n_iterations=300)
    assert actions.shape == (19, 1)           # pinned channel is not returned
    x = np.array([0.0, 0.0])
    for u in np.array(actions):
        x = A @ x + (B @ u).ravel() + c
    assert abs(x[0] - 1.0) < 0.2


# ============================================================================
# remember — posterior-carryover with exponential forgetting
# ============================================================================

def test_remember_accumulates_evidence_across_calls():
    """Two chunks with carryover: the second posterior is sharper than a
    single-chunk fit — evidence accumulates through the carried prior."""
    rng = np.random.RandomState(4)
    A = np.array([[0.95, 0.10], [-0.10, 0.92]])
    seqs = _drift_seqs(A, np.zeros(2), rng, n=16)

    single = LearnedLinear(dim=2, n_iterations=40)
    single.learn(seqs[:8])
    _, cov_single = gaussian_mean_cov(single.q_a)

    online = LearnedLinear(dim=2, n_iterations=40)
    online.learn(seqs[:8])
    online.remember()                         # forget=1: pure accumulation
    online.learn(seqs[8:])
    _, cov_online = gaussian_mean_cov(online.q_a)

    assert np.allclose(np.array(online.A), A, atol=0.04)
    assert float(jnp.trace(cov_online)) < 0.7 * float(jnp.trace(cov_single))


def test_forgetting_tracks_regime_change_where_full_memory_lags():
    """After A1 → A2, forget=0.2 lands on A2; forget=1 stays anchored between
    regimes. The stability–plasticity dial is one conjugate parameter."""
    A1 = np.array([[0.9, 0.0], [0.0, 0.9]])
    A2 = np.array([[0.5, 0.0], [0.0, 0.5]])

    def stream(forget):
        rng = np.random.RandomState(5)
        tr = LearnedLinear(dim=2, n_iterations=40)
        for regime, n_chunks in ((A1, 6), (A2, 3)):
            for _ in range(n_chunks):
                if tr.q_a is not None:
                    tr.remember(forget=forget)
                tr.learn(_drift_seqs(regime, np.zeros(2), rng, n=5))
        return float(np.linalg.norm(np.array(tr.A) - A2))

    err_forget, err_full = stream(0.2), stream(1.0)
    assert err_forget < 0.1
    assert err_forget < 0.5 * err_full


def test_remember_keeps_model_usable_between_chunks():
    """After remember(), the tempered posterior is the current belief —
    predict/plan work between chunks, and uncertainty has re-inflated."""
    rng = np.random.RandomState(6)
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    tr = LearnedLinear(dim=2, n_iterations=40)
    tr.learn(_drift_seqs(A, np.zeros(2), rng))
    _, cov_before = gaussian_mean_cov(tr.q_a)
    tr.remember(forget=0.5)
    assert tr.q_a is not None
    nxt = tr.predict(_msg(jnp.array([1.0, 1.0]), 1e4))
    assert np.allclose(np.array(gaussian_mean(nxt)), np.array(tr.A) @ np.ones(2), atol=0.05)
    _, cov_after = gaussian_mean_cov(tr.q_a)
    assert float(jnp.trace(cov_after)) > float(jnp.trace(cov_before))


def test_remember_diffuse_widens_only_the_drift():
    """`diffuse` is a random-walk step on c: its marginal variance grows by
    exactly the diffusion; B's marginals are untouched."""
    rng = np.random.RandomState(9)
    tr = LearnedLinear(dim=2, du=1, offset=True, n_iterations=20)
    seqs, ctrls = [], []
    for _ in range(10):
        us = [rng.randn(1) for _ in range(9)] + [np.zeros(1)]
        seqs.append(_drift_traj(np.eye(2) * 0.9, np.array([0.2, -0.1]), 10,
                                0.02, 0.01, rng, B=np.array([[0.3], [0.1]]), us=us))
        ctrls.append([jnp.asarray(u) for u in us[:9]])
    tr.learn(seqs, ctrls)
    _, Vb0 = gaussian_mean_cov(tr.q_b)
    tr.remember(diffuse=0.5)
    _, Vb1 = gaussian_mean_cov(tr.q_b)
    d, eff = 2, 2
    c_idx = [i * eff + (eff - 1) for i in range(d)]
    b_idx = [i * eff for i in range(d)]
    assert np.allclose(np.diag(np.array(Vb1))[c_idx],
                       np.diag(np.array(Vb0))[c_idx] + 0.5, atol=1e-5)
    assert np.allclose(np.diag(np.array(Vb1))[b_idx],
                       np.diag(np.array(Vb0))[b_idx], atol=1e-6)


def test_remember_diffuse_requires_offset():
    rng = np.random.RandomState(10)
    tr = LearnedLinear(dim=2, n_iterations=10)
    tr.learn(_drift_seqs(np.eye(2) * 0.9, np.zeros(2), rng, n=2))
    with pytest.raises(ValueError, match="offset=True"):
        tr.remember(diffuse=0.1)


def test_remember_validates_inputs():
    tr = LearnedLinear(dim=2)
    with pytest.raises(ValueError, match="call learn"):
        tr.remember()
    rng = np.random.RandomState(7)
    tr.learn(_drift_seqs(np.eye(2) * 0.9, np.zeros(2), rng, n=2))
    with pytest.raises(ValueError, match="forget must be"):
        tr.remember(forget=0.0)


def test_jointmodel_remember_delegates_to_learned_blocks():
    rng = np.random.RandomState(8)
    A = np.array([[0.9, 0.0], [0.0, 0.9]])
    raws = [[np.array(gaussian_mean(g)) for g in
             _drift_traj(A, np.zeros(2), 40, 0.02, 0.01, rng)]]
    block = Block("z", LearnedLinear(dim=2, n_iterations=20),
                  observe=lambda d: _msg(d, 1e4))
    model = JointModel([block])
    model.learn([{"z": r} for r in raws])
    model.remember(forget=0.5)
    assert block.transition._carried_a is not None
    assert block.transition.q_a is block.transition._carried_a
