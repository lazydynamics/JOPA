import numpy as np
import jax
import jax.numpy as jnp

from jopa.distributions import Gaussian, Wishart, gaussian_mean, gaussian_mean_cov, gaussian_prior
from jopa.message_passing import accumulate_vmp_messages
from jopa.nodes.transition import CTCache, CTMeta, ct_forward, ct_message_a


def _gaussian_from_mean_cov(mean, cov):
    mean = jnp.asarray(mean, dtype=float)
    cov = jnp.asarray(cov, dtype=float)
    lam = jnp.linalg.inv(cov)
    return Gaussian(eta=lam @ mean, lam=lam)


def _wishart_with_mean_precision(W):
    W = jnp.asarray(W, dtype=float)
    df = W.shape[0] + 3.0
    return Wishart(df=df, inv_scale=df * jnp.linalg.inv(W))


def test_ct_forward_matches_kalman_predict_step():
    A = jnp.array([[0.9, 0.2], [-0.1, 1.1]])
    Q = jnp.array([[0.04, 0.01], [0.01, 0.03]])
    W = jnp.linalg.inv(Q)
    mu = jnp.array([0.3, -0.7])
    cov = jnp.array([[0.5, 0.1], [0.1, 0.4]])
    q_a = gaussian_prior(4, 1e-12, A.ravel())
    q_W = _wishart_with_mean_precision(W)
    cache = CTCache(q_a, q_W, CTMeta(lambda a: a.reshape(2, 2)))

    predicted = ct_forward(_gaussian_from_mean_cov(mu, cov), cache)
    got_mu, got_cov = gaussian_mean_cov(predicted)
    expected_mu = A @ mu
    expected_cov = A @ cov @ A.T + Q

    assert np.allclose(np.asarray(got_mu), np.asarray(expected_mu), atol=1e-6)
    assert np.allclose(np.asarray(got_cov), np.asarray(expected_cov), atol=1e-6)


def test_accumulate_with_no_transitions_returns_parameter_prior():
    prior_a = gaussian_prior(4, 2.0, jnp.eye(2).ravel())
    prior_W = Wishart(df=5.0, inv_scale=2.0 * jnp.eye(2))
    cache = CTCache(prior_a, prior_W, CTMeta(lambda a: a.reshape(2, 2)))
    empty = Gaussian(eta=jnp.zeros((0, 2)), lam=jnp.zeros((0, 2, 2)))

    q_a, q_W, q_b = accumulate_vmp_messages(empty, empty, cache, None, prior_a, prior_W)

    assert q_b is None
    assert np.allclose(np.asarray(q_a.eta), np.asarray(prior_a.eta))
    assert np.allclose(np.asarray(q_a.lam), np.asarray(prior_a.lam))
    assert float(q_W.df) == float(prior_W.df)
    assert np.allclose(np.asarray(q_W.inv_scale), np.asarray(prior_W.inv_scale))


def test_message_a_posterior_collapses_to_scalar_mle_with_many_data():
    true_a = 1.7
    xs = np.linspace(-2.0, 2.0, 400)
    ys = true_a * xs
    W = jnp.array([[50.0]])
    q_a = gaussian_prior(1, 1e6)
    q_W = _wishart_with_mean_precision(W)
    cache = CTCache(q_a, q_W, CTMeta(lambda a: a.reshape(1, 1)))
    eta = q_a.eta
    lam = q_a.lam
    tiny_cov = 1e-9 * jnp.eye(2)
    for x, y in zip(xs, ys):
        q_yx = _gaussian_from_mean_cov(jnp.array([y, x]), tiny_cov)
        msg = ct_message_a(q_yx, cache)
        eta = eta + msg.eta
        lam = lam + msg.lam
    posterior = Gaussian(eta=eta, lam=lam)

    assert abs(float(gaussian_mean(posterior)[0]) - true_a) < 1e-5


def test_message_a_matches_finite_difference_quadratic_gradient():
    xs = jnp.array([-1.5, -0.25, 0.5, 1.25])
    ys = 1.3 * xs + 0.2
    precision = 7.0
    W = jnp.array([[precision]])
    q_a = gaussian_prior(1, 1e6)
    q_W = _wishart_with_mean_precision(W)
    cache = CTCache(q_a, q_W, CTMeta(lambda a: a.reshape(1, 1)))
    eta = jnp.zeros(1)
    lam = jnp.zeros((1, 1))
    tiny_cov = 1e-9 * jnp.eye(2)
    for x, y in zip(xs, ys):
        msg = ct_message_a(_gaussian_from_mean_cov(jnp.array([y, x]), tiny_cov), cache)
        eta = eta + msg.eta
        lam = lam + msg.lam

    def loss(a):
        residual = ys - a[0] * xs
        return 0.5 * precision * jnp.sum(residual ** 2)

    point = jnp.array([0.8])
    analytic_from_message = lam @ point - eta
    autodiff = jax.grad(loss)(point)
    eps = 1e-2
    finite_difference = jnp.array([(loss(point + eps) - loss(point - eps)) / (2 * eps)])

    assert np.allclose(np.asarray(analytic_from_message), np.asarray(autodiff), atol=1e-5)
    assert np.allclose(np.asarray(autodiff), np.asarray(finite_difference), atol=1e-3)
