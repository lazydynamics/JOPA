import importlib.util
from pathlib import Path

import numpy as np

from jopa.blocks import LearnedLinear

_PATH = Path(__file__).parents[1] / "examples" / "reacher" / "validation.py"
_SPEC = importlib.util.spec_from_file_location("reacher_pixel_validation", _PATH)
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


def test_state_free_rollout_gate_accepts_identified_linear_system():
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
    report = validation.rollout_diagnostics(model, encoded[8:], samples=64)
    gate = validation.dynamics_gate(report)
    assert report["6_step_nrmse"] < 0.5
    assert gate["passed"]


def test_gate_rejects_unidentified_control():
    report = {
        "one_step_nrmse": 0.1,
        "6_step_nrmse": 0.2,
        "mean_B_signal_to_std": 0.2,
    }
    gate = validation.dynamics_gate(report)
    assert not gate["passed"]
    assert not gate["checks"]["control_identified"]
