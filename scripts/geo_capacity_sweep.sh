#!/usr/bin/env bash
# Geometry/appearance balance sweep on SynthGeo (GT geometry + GT materials / relighting):
# geometry stage variant -> GT geometry metrics -> oracle-light material fit -> region relighting error.
#   bash scripts/geo_capacity_sweep.sh <gpu> <tag> "<train_oto args>" <scene> [<scene> ...]
# Results: output/oto/<tag>/<scene>/{geometry_gt.json, pbr_cap/metrics.json, attribution.txt}
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; TAG=$2; ARGS=$3; shift 3
source "$ROOT/scripts/env.sh"
for scene in "$@"; do
    M="$ROOT/output/oto/$TAG/$scene"
    run() { echo "=== [$TAG/$scene] $*"; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$@"; }
    if [ ! -f "$M/relight_mesh.pt" ]; then
        run "$ROOT/scripts/train_oto.py" --data "$ROOT/data/SynthGeo" --scene "$scene" --out "$M" \
            --init "mesh:$ROOT/output/synthgeo/$scene" $ARGS || continue
    fi
    run "$ROOT/scripts/eval_geometry_gt.py" -m "$M" --data "$ROOT/data/SynthGeo/$scene" > /dev/null   # writes geometry_gt.json
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --data "$ROOT/data/SynthGeo" --out pbr_cap
    run "$ROOT/scripts/attribute_error.py" --scene "$scene" --data "$ROOT/data/SynthGeo" --run pbr_cap \
        "$TAG=$M" > "$M/attribution.txt"
done
