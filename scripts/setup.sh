#!/usr/bin/env bash
# Build local extensions. Maintained by zhtang2004@gmail.com.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
# nvdiffrast is always built (ReMeS G-buffers, GLB baking). --with-tnt is accepted for
# backward compatibility and has no extra effect.
if [[ ${1:-} == --with-tnt ]]; then shift; fi
if (($#)); then echo 'Usage: bash scripts/setup.sh' >&2; exit 2; fi
if [[ -z ${CUDA_HOME:-} ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_HOME=$(dirname -- "$(dirname -- "$(command -v nvcc)")")
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA_HOME=/usr/local/cuda
    else
        echo 'Set CUDA_HOME to a CUDA toolkit containing bin/nvcc.' >&2; exit 1
    fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-4}
export SETUPTOOLS_USE_DISTUTILS=stdlib
MODULES=(diff-triangle-mesh-rasterization simple-knn effrdel nvdiffrast)
for MODULE in "${MODULES[@]}"; do
    echo "Building $MODULE"
    (cd -- "$ROOT/submodules/$MODULE" && "$PYTHON" setup.py build_ext --inplace)
done
# nvdiffrast imports its version from installed package metadata.
"$PYTHON" -m pip install --no-build-isolation --no-deps --no-cache-dir --no-compile "$ROOT/submodules/nvdiffrast"
echo 'Extensions built. Use scripts/run.sh to set the local import paths automatically.'
