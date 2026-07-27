import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "examples" / "reacher" / "validation.py"
_SPEC = importlib.util.spec_from_file_location("pixel_validation_full", _PATH)
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


def _passing_report():
    return {
        "one_step_nrmse": 0.10,
        "6_step_nrmse": 0.20,
        "mean_B_signal_to_std": 2.0,
        "pose_foreground_spearman": 0.80,
        "nearest_neighbor_visual_mismatch_p90": 0.10,
        "random_pair_visual_mismatch_p20": 0.30,
        "repeated_frame_motion_rms": 0.02,
        "zero_action_drift_20_nrmse": 0.05,
        "innovation_90_coverage": 0.90,
    }


def test_split_is_reproducible_and_trajectory_disjoint():
    trajectories = [object() for _ in range(20)]
    first = validation.split_trajectories(trajectories, seed=123)
    second = validation.split_trajectories(trajectories, seed=123)

    assert first["indices"] == second["indices"]
    index_sets = [
        set(first["indices"][name])
        for name in ("train", "calibration", "test")
    ]
    assert len(index_sets[0]) == 16
    assert len(index_sets[1]) == len(index_sets[2]) == 2
    assert not (index_sets[0] & index_sets[1])
    assert not (index_sets[0] & index_sets[2])
    assert not (index_sets[1] & index_sets[2])


@pytest.mark.parametrize(
    ("key", "value", "check"),
    [
        ("one_step_nrmse", 0.35, "one_step_latent_nrmse"),
        ("6_step_nrmse", 0.50, "six_step_latent_nrmse"),
        ("mean_B_signal_to_std", 1.0, "control_identified"),
        ("pose_foreground_spearman", 0.749, "pose_visual_distance"),
        (
            "nearest_neighbor_visual_mismatch_p90", 0.30,
            "visual_neighbour_aliasing",
        ),
        ("repeated_frame_motion_rms", 0.10, "repeated_frame_motion"),
        (
            "zero_action_drift_20_nrmse", 0.15,
            "zero_action_fixed_point",
        ),
        ("innovation_90_coverage", 0.84, "innovation_calibration"),
        ("innovation_90_coverage", 0.96, "innovation_calibration"),
    ],
)
def test_every_admission_failure_path(key, value, check):
    report = _passing_report()
    report[key] = value
    gate = validation.admission_gate(report)

    assert not gate["passed"]
    assert not gate["checks"][check]


def test_admission_accepts_all_predeclared_thresholds():
    gate = validation.admission_gate(_passing_report())
    assert gate["passed"]
    assert len(gate["checks"]) == 8
    assert all(gate["checks"].values())


def test_visual_metrics_distinguish_nearby_from_collapsed_latents():
    yy, xx = np.mgrid[:28, :28]
    positions = np.linspace(6.0, 22.0, 40)
    images = np.stack([
        np.exp(-((xx - position) ** 2 + (yy - 14) ** 2) / 45.0)
        for position in positions
    ]).astype(np.float32)
    normalized = (positions - positions.mean()) / positions.std()
    states = np.stack([
        normalized,
        normalized ** 2,
        np.sin(normalized),
        np.cos(normalized),
        np.gradient(normalized),
        np.zeros_like(normalized),
    ], axis=1)

    metrics = validation.visual_aliasing_metrics(
        images, states, pose_dim=4, background=np.zeros((28, 28)),
        pairs=4000, seed=9)
    collapsed = validation.visual_aliasing_metrics(
        images, np.zeros_like(states), pose_dim=4,
        background=np.zeros((28, 28)), pairs=4000, seed=9)
    identical = validation.foreground_mismatch(
        images[:3], images[:3], background=np.zeros((28, 28)))

    assert metrics["pose_foreground_spearman"] >= 0.75
    assert metrics["nearest_below_random"]
    assert collapsed["pose_foreground_spearman"] == 0.0
    assert not collapsed["nearest_below_random"]
    assert np.allclose(identical, 0.0)


class _Transition:
    def __init__(self, drift):
        self.A = np.eye(6)
        self.c = np.full(6, drift)


def test_zero_action_fixed_point_metric_detects_drift():
    rng = np.random.RandomState(12)
    reference = rng.randn(100, 6)
    rest = reference[:10]
    stable = validation.zero_action_fixed_point_drift(
        _Transition(0.0), rest, reference, steps=20)
    drifting = validation.zero_action_fixed_point_drift(
        _Transition(0.02), rest, reference, steps=20)

    assert stable == 0.0
    assert drifting > 0.15


def _manifest(tmp_path, passed=True):
    sensor = tmp_path / "sensor.msgpack"
    dynamics = tmp_path / "dynamics.pkl"
    sensor.write_bytes(b"sensor")
    dynamics.write_bytes(b"dynamics")
    gate = validation.admission_gate(_passing_report())
    gate["passed"] = passed
    return {
        "schema": validation.MANIFEST_SCHEMA,
        "architecture": {"kind": "test"},
        "split": {
            "seed": 4,
            "indices": {
                "train": [0, 1],
                "calibration": [2],
                "test": [3],
            },
        },
        "checkpoints": {
            "sensor": {
                "path": str(sensor),
                "sha256": validation.checkpoint_sha256(sensor),
            },
            "dynamics": {
                "path": str(dynamics),
                "sha256": validation.checkpoint_sha256(dynamics),
            },
        },
        "admission": gate,
    }


def test_manifest_refuses_failed_incompatible_and_stale_artifacts(tmp_path):
    manifest = _manifest(tmp_path)
    assert validation.validate_manifest(
        manifest, expected_architecture={"kind": "test"}) is manifest

    with pytest.raises(validation.ManifestError, match="architecture"):
        validation.validate_manifest(
            manifest, expected_architecture={"kind": "wrong"})

    failed = _manifest(tmp_path, passed=False)
    with pytest.raises(validation.ManifestError, match="did not pass"):
        validation.validate_manifest(failed)

    Path(manifest["checkpoints"]["sensor"]["path"]).write_bytes(b"changed")
    with pytest.raises(validation.ManifestError, match="hash mismatch"):
        validation.validate_manifest(manifest)

