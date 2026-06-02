import numpy as np
import jax
import jax.numpy as jnp

from jopa.nn.vae import VAE
from jopa.validate import (
    fit_linear_dynamics,
    predict_one_step,
    r2_score,
    validate_checkpoint,
)


def test_fit_linear_dynamics_recovers_controlled_map():
    A = np.array([[0.9, 0.2], [-0.1, 1.1]])
    B = np.array([[0.5], [-0.25]])
    x = np.array([0.2, -0.4])
    xs = [x]
    us = []
    for i in range(12):
        u = np.array([(-1) ** i * 0.3])
        us.append(u)
        x = A @ x + B @ u
        xs.append(x)
    A_hat, B_hat, pred = fit_linear_dynamics(np.asarray(xs), np.asarray(us))
    assert np.allclose(A_hat, A, atol=1e-8)
    assert np.allclose(B_hat, B, atol=1e-8)
    assert r2_score(np.asarray(xs)[1:], pred) > 0.999999


def test_predict_one_step_uses_checkpoint_dynamics():
    xs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    pred = predict_one_step(xs, A)
    assert np.allclose(pred, xs[:-1])


def test_validate_checkpoint_returns_finite_smoke_metrics():
    model = VAE(latent_dim=2)
    rng = jax.random.PRNGKey(0)
    params = model.init({"params": rng}, jnp.ones((1, 28, 28)), rng)
    observations = np.linspace(0.0, 1.0, 6 * 28 * 28, dtype=np.float32).reshape(6, 28, 28)
    report = validate_checkpoint(model, params, observations, max_reconstruction_mse=10.0)
    assert report.n_observations == 6
    assert report.latent_dim == 2
    assert report.dynamics_source == "fitted_latents"
    assert np.isfinite(report.reconstruction_mse)
    assert np.isfinite(report.one_step_latent_mse)
    assert report.passed
