import numpy as np

from examples.reacher import metrics
from jopa import LearnedLinear


def test_rollout_diagnostics_on_identified_linear_system():
    rng = np.random.RandomState(7)
    A = np.array([[1.0, 0.08], [0.0, 0.96]])
    B = np.array([[0.01], [0.15]])
    encoded = []
    for _ in range(10):
        x = rng.randn(2) * 0.2
        states, controls = [x.copy()], []
        for _ in range(50):
            u = rng.uniform(-1, 1, 1)
            x = A @ x + B @ u + rng.randn(2) * 0.002
            states.append(x.copy())
            controls.append(u)
        encoded.append((np.asarray(states), np.asarray(controls)))

    model = LearnedLinear(2, du=1, offset=True, n_iterations=12)
    model.learn_observed(
        [x for x, _ in encoded[:8]], [u for _, u in encoded[:8]],
        obs_prec=1e4)
    report = metrics.rollout_diagnostics(model, encoded[8:], samples=64)
    assert report["6_step_nrmse"] < 0.5
    assert report["mean_B_signal_to_std"] > 1.0
