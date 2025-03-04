#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <attention_matmul_spec_file> <input mlir> -o <output vmfb> [extra flags]

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

rm -rf "${SCRIPT_DIR}/configurations/punet"
rm -rf "${SCRIPT_DIR}/benchmarks/punet"

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-dump-executable-configurations-to="${SCRIPT_DIR}/configurations/punet" \
    --iree-hal-dump-executable-benchmarks-to="${SCRIPT_DIR}/benchmarks/punet" \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=$CHIP \
    --iree-execution-model=async-external \
    --iree-global-opt-propagate-transposes=1 \
    --iree-opt-const-eval=0 \
    --iree-opt-outer-dim-concat=1 \
    --iree-opt-aggressively-propagate-transposes=1 \
    --iree-dispatch-creation-enable-aggressive-fusion \
    --iree-hal-force-indirect-command-buffers \
    --iree-llvmgpu-enable-prefetch=1 \
    --iree-codegen-gpu-native-math-precision=1 \
    --iree-opt-data-tiling=0 \
    --iree-hal-memoization=1 \
    --iree-opt-strip-assertions \
    --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=1 \
    --iree-stream-resource-memory-model=discrete \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-config-add-tuner-attributes \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=0 \
    --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental),iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})' \
    "$@"
