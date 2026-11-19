#!/usr/bin/env bash
# Material-stage variants (train_pbr.py + Blender evaluation) on existing geometry runs.
#   bash scripts/material_runs.sh <gpu> <geometry tag> <run name> "<train_pbr args>" <scene> [<scene> ...]
# e.g. learned light + interreflection on v9:
#   bash scripts/material_runs.sh 0 v9 pbr_learned_ir "--learn-light --lr-light 5e-3 --face-res 4 \
#        --learn-normals --ssim 0.2 --interreflect" jugs hotdog
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-/media/mydrive1/miniconda3/envs/t2splat/bin/python}
GPU=$1; TAG=$2; RUN=$3; ARGS=$4; shift 4
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/oto_s4r/$TAG/$scene"
    echo "=== [$scene] $RUN: $ARGS"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" $ARGS --out "$RUN" &&
        CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run "$RUN"
    echo "=== [$scene] done"
done
