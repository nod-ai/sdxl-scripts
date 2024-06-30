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

readonly INPUT="$(realpath "$3")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

shift 3

set -x

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip="$CHIP" \
    --iree-rocm-bc-dir="${SCRIPT_DIR}/../bitcode-2024-03-07" \
    --iree-opt-const-expr-hoisting=false \
    --iree-opt-const-eval=false \
    --iree-opt-const-expr-hoisting=false \
    --iree-opt-data-tiling=false \
    --iree-flow-enable-aggressive-fusion \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-execution-model=async-external \
    "$@"
