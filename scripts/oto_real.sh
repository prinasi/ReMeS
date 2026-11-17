#!/usr/bin/env bash
# OTO-style geometry (Stage A soup + Stage B connected mesh) on Synthetic4Relight scenes, then the
# downstream pipeline: Phase 2 (oracle light, with Phase 4 one-bounce indirect light) and Phase 5 (Blender)
# on the new relight_mesh.pt; LEARN_LIGHT=1 adds Phase 3 (learned light) + Blender.
#   bash scripts/oto_real.sh <gpu> <tag> <scene> [<scene> ...]
# Extra train_oto.py arguments via OTO_ARGS; "@SCENE@" is replaced by the scene name, e.g.
#   OTO_ARGS="--init mesh:output/synthetic4relight/@SCENE@" bash scripts/oto_real.sh 0 v6 jugs
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; TAG=$2; shift 2
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/oto_s4r/$TAG/$scene"
    run() { echo "=== [$scene] $*"; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$@"; }
    run "$ROOT/scripts/train_oto.py" --data "$ROOT/data/Synthetic4Relight" --scene "$scene" --out "$M" ${OTO_ARGS//@SCENE@/$scene}
    run "$ROOT/scripts/eval_geometry.py" -m "$M" --scene "$scene" --relight-mesh || true
    run "$ROOT/scripts/diagnose_geometry.py" -m "$M" --scene "$scene"
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle
    run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_oracle
    if [ -n "${LEARN_LIGHT:-}" ]; then                      # Phase 3: learned training light
        run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --learn-light --out pbr_learned
        run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_learned
    fi
    echo "=== [$scene] done"
done
