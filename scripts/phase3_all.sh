#!/usr/bin/env bash
# Phase 3 driver: joint material + light optimisation (learned envmap) for scenes on one GPU.
#   bash scripts/phase3_all.sh <gpu> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; shift
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    echo "=== [$scene] learned light"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$ROOT/output/synthetic4relight/$scene" \
        --scene "$scene" --learn-light --out pbr_learned
    echo "=== [$scene] done"
done
