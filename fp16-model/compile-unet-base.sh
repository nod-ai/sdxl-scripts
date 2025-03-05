#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <default|winograd|misa> <attention_matmul_spec_file> <input mlir> -o <output vmfb> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi

readonly CHIP="$2"

readonly MODE="$3"
USE_WINOGRAD=0
USE_MISA=0
if [[ $MODE =~ "winograd" ]] ; then
  USE_WINOGRAD=1
elif [[ $MODE =~ "misa" ]] ; then
  USE_MISA=1
fi

readonly INPUT="$(realpath "$4")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

readonly SPEC_DIR="$(realpath "$SCRIPT_DIR"/specs)"
if [ ! -d "$SPEC_DIR" ] ; then
  echo "Spec directory not found: ${SPEC_DIR}"
  exit 1
fi

shift 4

# Removed since having this in causes issues on gfx11 as of last week
#readonly DEFAULT_FLAGS=(
#  "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)"
#)

set -x

# Note: --iree-dispatch-creation-enable-aggressive-fusion should be true
# but that was breaking gfx11 so it's temporarily removed

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-target-backends=rocm \
    --iree-hip-target="$CHIP" \
    --iree-hip-bc-dir="$(hipconfig --rocmpath)/amdgcn/bitcode" \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-const-eval=false \
    --iree-opt-data-tiling=false \
    --iree-hip-waves-per-eu=2 \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-dispatch-creation-enable-aggressive-fusion=false \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-execution-model=async-external \
    "$@"
