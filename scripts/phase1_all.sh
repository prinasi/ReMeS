#!/usr/bin/env bash
# Phase 1 driver: finish/run T²-Splat geometry + appearance for scenes on one GPU, then
# build the oriented relight mesh and its geometry report.
#   bash scripts/phase1_all.sh <gpu> <scene>[:texture] [<scene>[:texture] ...]
# ":texture" resumes a scene whose geometry is already trained.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; shift
source "$ROOT/scripts/env.sh"
OUT="$ROOT/output/synthetic4relight"
for item in "$@"; do
    scene=${item%%:*}
    extra=()
    [[ $item == *:texture ]] && extra=(--start-at texture)
    echo "=== [$scene] pipeline ${extra[*]:-}"
    "$PY" "$ROOT/scripts/run_pipeline.py" --scenes "$scene" --gpu "$GPU" "${extra[@]}"
    echo "=== [$scene] relight mesh"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/prepare_relight_mesh.py" -m "$OUT/$scene" --scene "$scene"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_geometry.py" -m "$OUT/$scene" --scene "$scene" --relight-mesh
    echo "=== [$scene] done"
done
