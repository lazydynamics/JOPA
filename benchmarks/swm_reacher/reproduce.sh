#!/usr/bin/env bash
# Every row of the result table on the eval seeds; one log per row and seed.
# Usage: reproduce.sh [row ...]      (default: all rows; `train` fits JOPA first)
#   nominal:  jopa  lewm
#   fault:    jopa_fault  jopa_frozen  lewm_frozen  lewm_finetune  lewm_gain_adam  lewm_gain_laplace
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUNS="${RUNS:-$HOME/jopa-bench/runs}"
SEEDS="${SEEDS:-42 1 2 3 4}"
JOPA="plan_config.horizon=3 plan_config.receding_horizon=1 +plan_config.history_len=3"
LEWM="objective=goal_mse_every plan_config.receding_horizon=1"
mkdir -p "$RUNS"

row() {   # row <name> <config> <overrides...>
    local name=$1 config=$2; shift 2
    for s in $SEEDS; do
        CONFIG=$config bash "$HERE/run_eval.sh" "$s" "$@" > "$RUNS/${name}_s$s.log" 2>&1
        echo "$name s$s $(grep -oE "success_rate.: [0-9.]+" "$RUNS/${name}_s$s.log")"
    done
}

run() {
    case $1 in
        train)             source ~/jopa-bench/.venv/bin/activate; cd "$HERE"
                           export STABLEWM_HOME=~/jopa-bench/swm_home XLA_PYTHON_CLIENT_PREALLOCATE=false
                           python encode.py --out "$RUNS/episodes.npz"
                           python train.py --episodes "$RUNS/episodes.npz" --out "${MODEL:-$RUNS/jopa_nl.pkl}" ;;
        jopa)              row jopa              reacher       solver=jopa_adaptive $JOPA ;;
        lewm)              row lewm              reacher       $LEWM ;;
        jopa_fault)        row jopa_fault        reacher_fault solver=jopa_adaptive $JOPA ;;
        jopa_frozen)       row jopa_frozen       reacher_fault solver=jopa $JOPA ;;
        lewm_frozen)       row lewm_frozen       reacher_fault $LEWM +plan_config.history_len=3 ;;
        lewm_finetune)     row lewm_finetune     reacher_fault solver=cem_online $LEWM +plan_config.history_len=3 ;;
        lewm_gain_adam)    row lewm_gain_adam    reacher_fault solver=cem_gain $LEWM +plan_config.history_len=3 ;;
        lewm_gain_laplace) row lewm_gain_laplace reacher_fault solver=cem_laplace_gain $LEWM +plan_config.history_len=3 ;;
        *) echo "unknown row $1" >&2; exit 1 ;;
    esac
}

for r in "${@:-jopa lewm jopa_fault jopa_frozen lewm_frozen lewm_finetune lewm_gain_adam lewm_gain_laplace}"; do
    for x in $r; do run "$x"; done
done
