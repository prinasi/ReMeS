#!/usr/bin/env bash
# ReMeS v1 relighting pipeline for scenes whose Phase 1 relight_mesh.pt exists:
#   Phase 2 (oracle light) -> Phase 3 (learned light) -> Phase 5 (GLB + Blender Cycles) for both.
#   bash scripts/relight_all.sh <gpu> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; shift
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/synthetic4relight/$scene"
    run() { echo "=== [$scene] $*"; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$@"; }
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_learned --learn-light
    run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_learned
    run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_oracle
    echo "=== [$scene] done"
done
