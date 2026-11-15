#!/usr/bin/env bash
# Train T²-Splat geometry on SynthGeo scenes with a config override, prepare the relight mesh and
# score it against GT geometry.
#   bash scripts/synthgeo_variant.sh <gpu> <override.yaml> <output_root> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; CFG=$2; OUT=$3; shift 3
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    echo "=== [$scene] geometry ($CFG)"
    "$PY" "$ROOT/scripts/run_pipeline.py" --dataset synthetic4relight --config "$CFG" --scenes "$scene" --gpu "$GPU"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/prepare_relight_mesh.py" -m "$OUT/$scene" --scene "$scene" \
        --data "$ROOT/data/SynthGeo"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_geometry_gt.py" -m "$OUT/$scene" --data "$ROOT/data/SynthGeo/$scene"
    if [[ -f "$OUT/$scene/training_snapshots.jsonl" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/analyze_rdt.py" -m "$OUT/$scene" --data "$ROOT/data/SynthGeo/$scene"
    fi
    echo "=== [$scene] done"
done
