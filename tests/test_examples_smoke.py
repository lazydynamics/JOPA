"""Smoke-run the digits example end to end in a tiny configuration.

The examples go through their real training, learning and planning paths; only
the sizes are cut, so a broken import or a broken sweep still fails here.
"""
import os
import subprocess
import sys
from pathlib import Path

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


def test_digits_rotating_smoke(tmp_path):
    _run("examples.digits_rotating", tmp_path)
