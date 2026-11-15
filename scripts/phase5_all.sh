#!/usr/bin/env bash
# Phase 5 driver: export each PBR run as GLB and relight it in Blender Cycles.
#   bash scripts/phase5_all.sh <gpu> <run> <scene> [<scene> ...]      (run: pbr_learned | pbr_oracle)
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; RUN=$2; shift 2
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    echo "=== [$scene/$RUN] Blender relighting"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_blender.py" -m "$ROOT/output/synthetic4relight/$scene" \
        --scene "$scene" --run "$RUN"
    echo "=== [$scene/$RUN] done"
done
