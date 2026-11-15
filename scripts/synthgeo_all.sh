#!/usr/bin/env bash
# D4 driver: render controlled scenes with GT geometry, train T²-Splat geometry, prepare relight mesh.
#   bash scripts/synthgeo_all.sh <gpu> <material> [<material> ...]     (diffuse | glossy | textured)
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; shift
source "$ROOT/scripts/env.sh"
for mat in "$@"; do
    scene="suzanne_$mat"
    D="$ROOT/data/SynthGeo/$scene"
    if [[ ! -f "$D/scene.json" ]]; then
        echo "=== [$scene] render"
        CUDA_VISIBLE_DEVICES=$GPU "$PY" -m relight.blender.make_synthetic --out "$D" --material "$mat" \
            --cameras "$ROOT/data/Synthetic4Relight/hotdog" --envmap "$ROOT/data/Synthetic4Relight/envmaps/envmap3.exr"
    fi
    echo "=== [$scene] T2-Splat geometry"
    "$PY" "$ROOT/scripts/run_pipeline.py" --dataset synthetic4relight --config "$ROOT/configs/overrides/synthgeo.yaml" \
        --scenes "$scene" --gpu "$GPU"
    echo "=== [$scene] relight mesh"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/prepare_relight_mesh.py" -m "$ROOT/output/synthgeo/$scene" \
        --scene "$scene" --data "$ROOT/data/SynthGeo"
    echo "=== [$scene] done"
done
