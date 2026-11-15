#!/usr/bin/env bash
# D3 driver: Phase 2 (oracle) + Blender evaluation on geometry variants.
#   bash scripts/geomvar_eval.sh <gpu> <variant> <vis_tmin> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; VAR=$2; TMIN=$3; shift 3
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/geomvar/$scene/$VAR"
    RUN="pbr_oracle_t$TMIN"
    echo "=== [$scene/$VAR] oracle, vis_tmin $TMIN"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out "$RUN" --vis-tmin "$TMIN"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run "$RUN"
    echo "=== [$scene/$VAR] done"
done
