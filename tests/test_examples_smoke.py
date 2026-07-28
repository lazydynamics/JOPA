"""Smoke-run every non-Reacher example end to end in a tiny configuration.

The examples go through their real training, learning and planning paths; only
the sizes are cut, so a broken import or a broken sweep still fails here.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(module: str, tmp_path: Path, *args: str):
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", module, "--smoke",
         "--checkpoint-dir", str(tmp_path / "checkpoints"),
         "--output-dir", str(tmp_path / "outputs"), *args],
        cwd=ROOT, env=env, check=True, timeout=600)


@pytest.mark.parametrize("module", [
    "examples.digits_rotating",
    "examples.digits_controlled",
    "examples.digits_end_to_end",
])
def test_digits_example_smoke(module, tmp_path):
    _run(module, tmp_path)


def test_pendulum_smoke(tmp_path):
    _run("examples.pendulum", tmp_path, "--no-cache", "--n-frames", "2",
         "--horizon", "2", "--exec-steps", "1", "--n-replans", "1")
