#!/usr/bin/env bash
# Re-run Stage B (from a saved Stage A soup) + downstream Phase 2 / Phase 5 on Synthetic4Relight scenes.
#   bash scripts/oto_real_b.sh <gpu> <src_tag> <dst_tag> <scene> [<scene> ...] [-- extra train_oto.py args]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
GPU=$1; SRC=$2; DST=$3; shift 3
scenes=(); extra=()
while (($#)); do if [[ $1 == -- ]]; then shift; extra=("$@"); break; fi; scenes+=("$1"); shift; done
source "$ROOT/scripts/env.sh"
for scene in "${scenes[@]}"; do
    M="$ROOT/output/oto_s4r/$DST/$scene"
    run() { echo "=== [$scene] $*"; CUDA_VISIBLE_DEVICES=$GPU "$PY" "$@"; }
    run "$ROOT/scripts/train_oto.py" --data "$ROOT/data/Synthetic4Relight" --scene "$scene" --out "$M" \
        --resume "$ROOT/output/oto_s4r/$SRC/$scene" "${extra[@]}"
    run "$ROOT/scripts/eval_geometry.py" -m "$M" --scene "$scene" --relight-mesh || true
    run "$ROOT/scripts/train_pbr.py" -m "$M" --scene "$scene" --out pbr_oracle
    run "$ROOT/scripts/eval_blender.py" -m "$M" --scene "$scene" --run pbr_oracle
    echo "=== [$scene] done"
done
