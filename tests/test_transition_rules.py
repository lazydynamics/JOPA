"""Each ContinuousTransition rule checked against an independent reference:
a dense Gaussian solve, autodiff of the expected energy, or Monte Carlo.

The node's averaged factor, under q(a) q(b) q(W), is

    exp E[log f] ∝ exp -½[(y − mA x − mB u)ᵀ mW (y − mA x − mB u) + xᵀ C x],

with C the Va contraction. Every rule must be a message of this one factor.
"""
import jax
import jax.numpy as jnp
import numpy as np

from jopa.distributions import Gaussian, Wishart, gaussian_mean_cov
from jopa.nodes.transition import (
    CTCache,
    CTMeta,
    _va_contraction,
    ct_backward,
    ct_forward,
    ct_marginal_yx,
    ct_message_a,
    ct_message_b,
    ct_message_W,
    ct_moments,
    ct_parameter_messages,
    ct_predict,
)
from jopa.transitions import _chain

D, DU = 2, 1
A_TRUE = np.array([[0.95, 0.15], [-0.1, 0.9]])
B_TRUE = np.array([[0.2], [0.5]])


def _gauss(mean, cov):
    lam = jnp.linalg.inv(jnp.asarray(cov, dtype=jnp.float32))
    return Gaussian(eta=lam @ jnp.asarray(mean, dtype=jnp.float32), lam=lam)


def _cache(a_var=0.02, b_var=0.05, df=12.0):
    """A cache with parameter uncertainty large enough to matter."""
    rng = np.random.RandomState(0)
    Ma = rng.randn(D * D, D * D)
    Va = a_var * (np.eye(D * D) + 0.3 * Ma @ Ma.T / (D * D))
    Vb = b_var * np.eye(D * DU)
    q_a = _gauss(A_TRUE.ravel(), Va)
    q_b = _gauss(B_TRUE.ravel(), Vb)
    S = np.array([[0.4, 0.1], [0.1, 0.3]]) * df / 10.0
    q_W = Wishart(df=df, inv_scale=jnp.asarray(S, dtype=jnp.float32))
    return CTCache(q_a, q_W, CTMeta(lambda a: a.reshape(D, D)), q_b)


def _np(c):
    def f(v):
        return np.asarray(v, dtype=np.float64)
    C = f(_va_contraction(c.mW, c.Va, D, D))
    return f(c.mA), f(c.mB), f(c.mW), C


# --------------------------------------------------------------------------
# State messages — all three must come from the same averaged factor
# --------------------------------------------------------------------------

def test_chain_sweep_equals_dense_vmp_posterior_and_the_ct_messages():
    """Eliminating the chain (`jopa.fg`) gives the dense solve of the
    averaged-factor chain, on non-centred latents where the C term matters,
    and its pairwise marginals are the CT joint rule fed by the CT forward and
    backward messages."""
    c = _cache()
    A, B, W, C = _np(c)
    T = 6
    rng = np.random.RandomState(1)
    means = 3.0 + rng.randn(T, D)
    obs_prec = 4.0
    obs = [_gauss(m, np.eye(D) / obs_prec) for m in means]
    us = [jnp.array([0.3 * np.sin(t)], dtype=jnp.float32) for t in range(T - 1)]
    prior_x = Gaussian(eta=jnp.zeros(D), lam=jnp.eye(D))

    stacked = Gaussian(jnp.stack([g.eta for g in obs]), jnp.stack([g.lam for g in obs]))
    got_mean, got_cov, got_cross = _chain(stacked, c, jnp.stack(us))

    P = np.zeros((T * D, T * D))
    h = np.zeros(T * D)
    def blk(t):
        return slice(t * D, (t + 1) * D)
    P[blk(0), blk(0)] += np.eye(D)
    for t in range(T):
        P[blk(t), blk(t)] += obs_prec * np.eye(D)
        h[blk(t)] += obs_prec * means[t]
    for t in range(T - 1):
        Bu = B @ np.asarray(us[t])
        P[blk(t + 1), blk(t + 1)] += W
        P[blk(t), blk(t)] += A.T @ W @ A + C
        P[blk(t + 1), blk(t)] += -W @ A
        P[blk(t), blk(t + 1)] += -A.T @ W
        h[blk(t + 1)] += W @ Bu
        h[blk(t)] += -A.T @ W @ Bu
    cov = np.linalg.inv(P)
    mean = cov @ h

    for t in range(T):
        assert np.allclose(got_mean[t], mean[blk(t)], atol=2e-3)
        assert np.allclose(got_cov[t], cov[blk(t), blk(t)], atol=2e-4)
    for t in range(T - 1):
        assert np.allclose(got_cross[t], cov[blk(t), blk(t + 1)], atol=2e-4)

    def times(g, h):
        return Gaussian(g.eta + h.eta, g.lam + h.lam)
    alpha = [prior_x]
    for t in range(T - 1):
        alpha.append(ct_forward(times(alpha[t], obs[t]), c, us[t]))
    beta = [Gaussian(jnp.zeros(D), jnp.zeros((D, D)))]
    for t in range(T - 2, -1, -1):
        beta.insert(0, ct_backward(times(beta[0], obs[t + 1]), c, us[t]))
    for t in range(T - 1):
        q = ct_marginal_yx(times(beta[t + 1], obs[t + 1]), times(alpha[t], obs[t]), c, us[t])
        m, V = gaussian_mean_cov(q)
        assert np.allclose(m, np.concatenate([got_mean[t + 1], got_mean[t]]), atol=2e-3)
        assert np.allclose(V[D:, :D], got_cross[t], atol=2e-4)


def test_joint_marginal_is_consistent_with_forward_and_backward():
    """The y- and x-marginals of q(y, x) are m_y·forward(m_x) and m_x·backward(m_y)."""
    c = _cache()
    u = jnp.array([0.7])
    m_x = _gauss([2.0, -1.0], [[0.3, 0.05], [0.05, 0.2]])
    m_y = _gauss([2.5, -0.5], [[0.5, 0.0], [0.0, 0.4]])

    joint_mean, joint_cov = gaussian_mean_cov(ct_marginal_yx(m_y, m_x, c, u))
    fwd = ct_forward(m_x, c, u)
    bwd = ct_backward(m_y, c, u)
    y_mean, y_cov = gaussian_mean_cov(
        Gaussian(m_y.eta + fwd.eta, m_y.lam + fwd.lam))
    x_mean, x_cov = gaussian_mean_cov(
        Gaussian(m_x.eta + bwd.eta, m_x.lam + bwd.lam))

    assert np.allclose(joint_mean[:D], y_mean, atol=1e-4)
    assert np.allclose(joint_cov[:D, :D], y_cov, atol=1e-5)
    assert np.allclose(joint_mean[D:], x_mean, atol=1e-4)
    assert np.allclose(joint_cov[D:, D:], x_cov, atol=1e-5)


# --------------------------------------------------------------------------
# Parameter messages — natural parameters of the expected energy
# --------------------------------------------------------------------------

def _q_yx():
    mean = jnp.array([1.4, -0.6, 1.0, -0.8])
    L = jnp.array([[0.3, 0, 0, 0], [0.05, 0.25, 0, 0],
                   [0.1, 0.02, 0.2, 0], [0.0, 0.08, 0.04, 0.3]])
    return _gauss(mean, L @ L.T), mean, L @ L.T


def _energy(Amat, Bmat, W, mean, cov, u):
    """E_q(y,x)[½ (y − A x − B u)ᵀ W (y − A x − B u)] for fixed A, B."""
    M = jnp.concatenate([jnp.eye(D), -Amat], axis=1)         # r = M [y; x] − B u
    r = M @ mean - Bmat @ u
    return 0.5 * (r @ W @ r + jnp.trace(W @ M @ cov @ M.T))


def test_message_a_is_the_expected_energy_in_a():
    c = _cache()
    q, mean, cov = _q_yx()
    u = jnp.array([0.4])
    msg = ct_message_a(q, c, u)
    def f(a):
        return _energy(a.reshape(D, D), c.mB, c.mW, mean, cov, u)
    a0 = jnp.zeros(D * D)
    assert np.allclose(msg.lam, jax.hessian(f)(a0), atol=1e-4)
    assert np.allclose(msg.eta, -jax.grad(f)(a0), atol=1e-4)


def test_message_b_is_the_expected_energy_in_b():
    c = _cache()
    q, mean, cov = _q_yx()
    u = jnp.array([0.4])
    msg = ct_message_b(q, c, u)
    def f(b):
        return _energy(c.mA, b.reshape(D, DU), c.mW, mean, cov, u)
    b0 = jnp.zeros(D * DU)
    assert np.allclose(msg.lam, jax.hessian(f)(b0), atol=1e-4)
    assert np.allclose(msg.eta, -jax.grad(f)(b0), atol=1e-4)


def test_message_W_is_the_expected_residual_scatter():
    """Δ = E[(y − Ax − Bu)(y − Ax − Bu)ᵀ] under q(y,x) q(a) q(b), by Monte Carlo."""
    c = _cache()
    q, mean, cov = _q_yx()
    u = np.array([0.4])
    rng = np.random.default_rng(0)
    n = 400_000
    yx = rng.multivariate_normal(np.asarray(mean), np.asarray(cov), n)
    a = rng.multivariate_normal(np.asarray(c.ma), np.asarray(c.Va), n)
    b = rng.multivariate_normal(np.asarray(c.mb), np.asarray(c.Vb), n)
    r = (yx[:, :D] - np.einsum("nij,nj->ni", a.reshape(n, D, D), yx[:, D:])
         - np.einsum("nij,j->ni", b.reshape(n, D, DU), u))
    mc = r.T @ r / n
    got = np.asarray(ct_message_W(q, c, jnp.asarray(u)))
    assert np.allclose(got, mc, rtol=2e-2, atol=5e-3)


# --------------------------------------------------------------------------
# Predictive — averages the transition, not its log
# --------------------------------------------------------------------------

def test_predict_matches_monte_carlo_predictive():
    c = _cache()
    u = np.array([0.6])
    mx = np.array([2.0, -1.5])
    Vx = np.array([[0.2, 0.05], [0.05, 0.1]])
    rng = np.random.default_rng(1)
    n = 400_000
    x = rng.multivariate_normal(mx, Vx, n)
    a = rng.multivariate_normal(np.asarray(c.ma), np.asarray(c.Va), n)
    b = rng.multivariate_normal(np.asarray(c.mb), np.asarray(c.Vb), n)
    # W ~ Wishart(ν, S⁻¹) as a sum of ν outer products (ν is an integer here)
    S = np.asarray(c.mW_inv) * 12.0
    g = rng.multivariate_normal(np.zeros(D), np.linalg.inv(S), (n, 12))
    W = np.einsum("nki,nkj->nij", g, g)
    noise = np.einsum("nij,nj->ni", np.linalg.cholesky(np.linalg.inv(W)),
                      rng.standard_normal((n, D)))
    y = (np.einsum("nij,nj->ni", a.reshape(n, D, D), x)
         + np.einsum("nij,j->ni", b.reshape(n, D, DU), u) + noise)

    got_mean, got_cov = gaussian_mean_cov(
        ct_predict(_gauss(mx, Vx), c, jnp.asarray(u)))
    assert np.allclose(got_mean, y.mean(0), atol=1e-2)
    assert np.allclose(got_cov, np.cov(y.T), rtol=3e-2, atol=1e-2)


def test_predict_reduces_to_kalman_when_parameters_are_known():
    c = _cache(a_var=1e-9, b_var=1e-9, df=1e7)
    A, B, W, _ = _np(c)
    mx, Vx = np.array([0.3, -0.7]), np.array([[0.5, 0.1], [0.1, 0.4]])
    u = np.array([0.2])
    got_mean, got_cov = gaussian_mean_cov(
        ct_predict(_gauss(mx, Vx), c, jnp.asarray(u)))
    assert np.allclose(got_mean, A @ mx + B @ u, atol=1e-5)
    assert np.allclose(got_cov, A @ Vx @ A.T + np.linalg.inv(W), atol=1e-5)


# --------------------------------------------------------------------------
# Batched parameter update — summed moments equal the summed per-step rules
# --------------------------------------------------------------------------

def test_summed_moments_equal_sum_of_per_step_messages():
    c = _cache()
    rng = np.random.RandomState(3)
    n = 7
    m_xs = [_gauss(1.0 + rng.randn(D), np.eye(D) * 0.2) for _ in range(n)]
    m_ys = [_gauss(1.0 + rng.randn(D), np.eye(D) * 0.3) for _ in range(n)]
    us = [jnp.asarray(rng.randn(DU), dtype=jnp.float32) for _ in range(n)]
    prior_a = Gaussian(eta=jnp.zeros(D * D), lam=jnp.eye(D * D))
    prior_b = Gaussian(eta=jnp.zeros(D * DU), lam=jnp.eye(D * DU))
    prior_W = Wishart(df=5.0, inv_scale=jnp.eye(D))

    joints = [ct_marginal_yx(m_y, m_x, c, u) for m_x, m_y, u in zip(m_xs, m_ys, us, strict=True)]
    stats = jax.tree.map(lambda *v: sum(v), *[ct_moments(q, c, u) for q, u in zip(joints, us, strict=True)])
    msg_a, delta, msg_b = ct_parameter_messages(stats, c)
    q_a, q_b = prior_a * msg_a, prior_b * msg_b
    q_W = Wishart(prior_W.df + stats.n, prior_W.inv_scale + delta)

    eta_a, lam_a = prior_a.eta, prior_a.lam
    eta_b, lam_b = prior_b.eta, prior_b.lam
    S = prior_W.inv_scale
    for m_x, m_y, u in zip(m_xs, m_ys, us, strict=True):
        q = ct_marginal_yx(m_y, m_x, c, u)
        a, b = ct_message_a(q, c, u), ct_message_b(q, c, u)
        eta_a, lam_a = eta_a + a.eta, lam_a + a.lam
        eta_b, lam_b = eta_b + b.eta, lam_b + b.lam
        S = S + ct_message_W(q, c, u)

    assert np.allclose(q_a.eta, eta_a, rtol=1e-4, atol=1e-3)
    assert np.allclose(q_a.lam, lam_a, rtol=1e-4, atol=1e-3)
    assert np.allclose(q_b.eta, eta_b, rtol=1e-4, atol=1e-3)
    assert np.allclose(q_b.lam, lam_b, rtol=1e-4, atol=1e-3)
    assert np.allclose(q_W.inv_scale, S, rtol=1e-4, atol=1e-3)
    assert float(q_W.df) == prior_W.df + n
