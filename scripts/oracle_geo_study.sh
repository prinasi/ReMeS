#!/usr/bin/env bash
# Oracle-geometry ablation on SynthGeo: the same Phase 2 material fit (oracle light) on
# GT geometry vs reconstructed geometries; the metric gap is the cost of geometry error.
#   bash scripts/oracle_geo_study.sh <gpu> <scene> <name>=<model dir> [...]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; SCENE=$2; shift 2
source "$ROOT/scripts/env.sh"
for spec in "$@"; do
    name=${spec%%=*}; M=${spec#*=}
    echo "=== [$SCENE/$name] $M"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$SCENE" \
        --data "$ROOT/data/SynthGeo" --out pbr_oraclegeo
done
