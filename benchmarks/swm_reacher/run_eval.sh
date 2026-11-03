#!/usr/bin/env bash
# Upstream eval_wm.py, unmodified; extra solvers come from this directory.
# Usage: [CONFIG=reacher_fault] run_eval.sh <seed> [hydra overrides...]   e.g.  run_eval.sh 42 solver=jopa
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source ~/jopa-bench/.venv/bin/activate
export STABLEWM_HOME=~/jopa-bench/swm_home MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
cd ~/jopa-bench/stable-worldmodel/scripts/plan
seed="$1"; shift
python -u eval_wm.py --config-dir="$HERE/config" --config-name="${CONFIG:-reacher}" \
    policy=quentinll/lewm-reacher seed="$seed" "$@"
