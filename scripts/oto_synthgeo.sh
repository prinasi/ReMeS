#!/usr/bin/env bash
# Stage A (OTO-style opaque soup) on SynthGeo scenes + GT geometry evaluation.
#   bash scripts/oto_synthgeo.sh <gpu> <tag> <scene> [<scene> ...] [-- extra train_oto.py args]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; TAG=$2; shift 2
scenes=(); extra=()
while (($#)); do if [[ $1 == -- ]]; then shift; extra=("$@"); break; fi; scenes+=("$1"); shift; done
source "$ROOT/scripts/env.sh"
for scene in "${scenes[@]}"; do
    OUT="$ROOT/output/oto/$TAG/$scene"
    echo "=== [$TAG/$scene] train_oto ${extra[*]:-}"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/train_oto.py" --data "$ROOT/data/SynthGeo" --scene "$scene" --out "$OUT" "${extra[@]}"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/eval_geometry_gt.py" -m "$OUT" --data "$ROOT/data/SynthGeo/$scene"
    echo "=== [$TAG/$scene] done"
done
