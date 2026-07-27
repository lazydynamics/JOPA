import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jopa import PoseMotionObservation
from jopa.distributions import gaussian_mean_cov
from jopa.nn.vae import PoseMotionVAE, load_params, save_params


def _sensor():
    model = PoseMotionVAE(
        pose_dim=4, motion_dim=2, ch=4, n_frames=4,
        img_size=28, motion_hidden_dim=8)
    key = jax.random.PRNGKey(17)
    params = model.init(
        {"params": key}, jnp.ones((1, 4, 28, 28)), key)
    return model, params, PoseMotionObservation(model, params)


def test_pose_motion_shapes_structure_and_gaussian_message():
    _, _, observation = _sensor()
    rng = np.random.RandomState(1)
    window = rng.rand(4, 28, 28).astype(np.float32)
    alternate = rng.rand(4, 28, 28).astype(np.float32)
    alternate[-1] = window[-1]

    mean, log_std = observation.encode(window)
    other, _ = observation.encode(alternate)
    message = observation.message(window)
    message_mean, covariance = gaussian_mean_cov(message)

    assert mean.shape == log_std.shape == (6,)
    assert message.eta.shape == (6,)
    assert message.lam.shape == covariance.shape == (6, 6)
    assert np.all(np.diag(np.asarray(covariance)) > 0)
    assert np.allclose(np.asarray(message_mean), np.asarray(mean))
    assert np.allclose(np.asarray(mean[:4]), np.asarray(other[:4]), atol=1e-6)


def test_repeated_frames_are_exact_zero_motion_and_static_goal():
    _, _, observation = _sensor()
    frame = np.random.RandomState(2).rand(28, 28).astype(np.float32)
    repeated = np.repeat(frame[None], 4, axis=0)
    state, _ = observation.encode(repeated)
    goal, _ = observation.encode(frame)

    assert np.allclose(np.asarray(state[4:]), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(goal[4:]), 0.0, atol=1e-7)
    assert np.allclose(np.asarray(goal[:4]), np.asarray(state[:4]), atol=1e-6)


def test_decoder_is_pose_only():
    _, _, observation = _sensor()
    first = jnp.array([0.1, -0.2, 0.3, -0.4, 8.0, -9.0])
    second = first.at[4:].set(jnp.array([-2.0, 3.0]))
    decoded_first = observation.decode(first)
    decoded_second = observation.decode(second)

    assert decoded_first.shape == (28, 28)
    assert np.allclose(
        np.asarray(decoded_first), np.asarray(decoded_second), atol=1e-7)


def test_decoder_jacobian_goal_precision_is_block_psd():
    _, _, observation = _sensor()
    yy, xx = np.mgrid[:28, :28]
    goal = np.exp(
        -((xx - 17) ** 2 + (yy - 13) ** 2) / 12
    ).astype(np.float32)
    precision = np.asarray(observation.goal_precision(
        goal, background=np.zeros_like(goal)))

    assert precision.shape == (6, 6)
    assert np.allclose(precision, precision.T, atol=1e-5)
    assert np.linalg.eigvalsh(precision).min() >= -1e-5
    assert np.isclose(np.diag(precision[:4, :4]).mean(), 50.0, rtol=1e-4)
    assert np.allclose(precision[4:, 4:], 100.0 * np.eye(2))
    assert np.allclose(precision[:4, 4:], 0.0)
    assert np.allclose(precision[4:, :4], 0.0)


def test_pose_motion_checkpoint_roundtrip_and_incompatibility(tmp_path):
    model, params, observation = _sensor()
    path = tmp_path / "sensor.msgpack"
    save_params(params, path)
    restored = load_params(model, path)
    restored_observation = PoseMotionObservation(model, restored)
    frame = np.ones((4, 28, 28), dtype=np.float32)
    assert np.allclose(
        np.asarray(observation.encode(frame)[0]),
        np.asarray(restored_observation.encode(frame)[0]))

    incompatible = PoseMotionVAE(
        pose_dim=3, motion_dim=2, ch=4, n_frames=4,
        img_size=28, motion_hidden_dim=8)
    with pytest.raises(Exception):
        load_params(incompatible, path)


def test_per_dimension_calibration_offsets_match_ninety_percent():
    rng = np.random.RandomState(5)
    base_log_std = np.log(np.full((4000, 6), 0.25))
    scale = np.linspace(0.5, 2.0, 6)
    innovations = rng.randn(4000, 6) * scale
    offsets = PoseMotionObservation.calibration_offsets(
        innovations, base_log_std, coverage=0.90)
    calibrated_std = np.exp(base_log_std + offsets)
    inside = np.abs(innovations) <= 1.6448536269514722 * calibrated_std

    assert offsets.shape == (6,)
    assert np.all(np.isfinite(offsets))
    assert np.allclose(inside.mean(axis=0), 0.90, atol=0.01)

