#!/usr/bin/env bash
# One-time environment for the JOPA vs LeWM Reacher benchmark.
# Everything is created under $BENCH (default ~/jopa-bench); nothing else is touched.
# Usage (on the GPU machine, from a copy of this repo):  bash benchmarks/swm_reacher/setup.sh
set -euo pipefail

LEWM_COMMIT=8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
SWM_COMMIT=21446f1ede6d5284e981bd7b47f432b994e6d812
JOPA_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="${BENCH:-$HOME/jopa-bench}"
export STABLEWM_HOME="$BENCH/swm_home"
mkdir -p "$BENCH" "$STABLEWM_HOME"
cd "$BENCH"

pin() {  # pin <url> <dir> <commit>
    [ -d "$2" ] || git clone -q "$1" "$2"
    git -C "$2" fetch -q origin "$3" && git -C "$2" checkout -q "$3"
}
pin https://github.com/lucas-maes/le-wm.git le-wm "$LEWM_COMMIT"
pin https://github.com/galilai-group/stable-worldmodel.git stable-worldmodel "$SWM_COMMIT"

[ -d .venv ] || uv venv -q --python 3.10 .venv
source .venv/bin/activate
uv pip install -q --torch-backend=auto torch torchvision
# transformers 5 renamed the ViT parameters; the released checkpoint (2026-03)
# uses the 4.x names, so the whole set is resolved against transformers<5.
uv pip install -q -e "./stable-worldmodel[train,env,format]" huggingface_hub "transformers<5"
uv pip install -q -e "$JOPA_DIR" "jax[cuda12]==0.6.2"   # JAX beside torch on the GPU (run_eval.sh disables preallocation)

python - <<'PY'
import torch, jax
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch cannot see the GPU"
print(torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda"); torch.cuda.synchronize(); (x @ x).sum().item()
print("jax", jax.__version__, jax.devices())
PY

# Released LeWM Reacher dataset (MIT), placed where eval_wm.py's reacher config
# looks for it. The release is LeWM's random-policy Reacher data; the config
# names it dmc/reacher_random.h5. Reproducing LeWM's number checks the match.
DATA="$STABLEWM_HOME/datasets/dmc/reacher_random.h5"
if [ ! -f "$DATA" ]; then
    hf download quentinll/lewm-reacher --repo-type dataset --local-dir "$BENCH/download"
    mkdir -p "$(dirname "$DATA")"
    tar --zstd -xf "$BENCH/download/reacher.tar.zst" -C "$BENCH/download"
    mv "$BENCH/download/reacher.h5" "$DATA"
    rm -rf "$BENCH/download"
fi

# Released LeWM checkpoint: load_pretrained builds it from the Hub repo's
# Hydra config.json + weights.pt and caches it under $STABLEWM_HOME/checkpoints.
(cd /tmp && python - <<'PY'
import stable_worldmodel as swm
model = swm.wm.utils.load_pretrained("quentinll/lewm-reacher")
print("loaded", type(model).__name__, sum(p.numel() for p in model.parameters()), "params")
data = swm.data.load_dataset("dmc/reacher_random.h5", keys_to_cache=["action"])
print("dataset", type(data).__name__, data.column_names)
PY
)

du -sh "$STABLEWM_HOME"/*
echo "le-wm $(git -C le-wm rev-parse --short HEAD)  stable-worldmodel $(git -C stable-worldmodel rev-parse --short HEAD)"
echo SETUP_DONE
