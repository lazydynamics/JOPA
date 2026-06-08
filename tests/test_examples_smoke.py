import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_example(script: str, tmp_path: Path, *args: str):
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(ROOT / "examples" / script),
        "--smoke",
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--output-dir",
        str(tmp_path / "outputs"),
        *args,
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=180)


def test_rotating_digits_smoke(tmp_path):
    _run_example("rotating_digits.py", tmp_path)


def test_controlled_digits_smoke(tmp_path):
    _run_example("controlled_digits.py", tmp_path)


def test_end_to_end_digits_smoke(tmp_path):
    _run_example("end_to_end_digits.py", tmp_path)


def test_pendulum_smoke(tmp_path):
    _run_example(
        "pendulum.py",
        tmp_path,
        "--no-cache",
        "--n-frames",
        "2",
        "--horizon",
        "2",
        "--exec-steps",
        "1",
        "--n-replans",
        "1",
    )
