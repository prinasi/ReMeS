#!/usr/bin/env bash
# Phase 2 driver: oracle-light PBR materials (+ no-visibility ablation) for scenes on one GPU.
#   bash scripts/phase2_all.sh <gpu> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; shift
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/synthetic4relight/$scene"
    echo "=== [$scene] oracle light"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle
    echo "=== [$scene] oracle light, no visibility (ablation)"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle_novis --no-visibility
    echo "=== [$scene] done"
done
