#!/usr/bin/env bash
# OTO-style geometry (Stage A soup + Stage B connected mesh) on Synthetic4Relight scenes, then the
# downstream pipeline: Phase 2 (oracle light) and Phase 5 (Blender) on the new relight_mesh.pt.
#   bash scripts/oto_real.sh <gpu> <tag> <scene> [<scene> ...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; TAG=$2; shift 2
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/oto_s4r/$TAG/$scene"
    run() { echo "=== [$scene] $*"; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$@"; }
    run "$ROOT/scripts/train_oto.py" --data "$ROOT/data/Synthetic4Relight" --scene "$scene" --out "$M"
    run "$ROOT/scripts/eval_geometry.py" -m "$M" --scene "$scene" --relight-mesh || true
    run "$ROOT/scripts/diagnose_geometry.py" -m "$M" --scene "$scene"
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle
    run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_oracle
    echo "=== [$scene] done"
done
