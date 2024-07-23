#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}
shift

TRANSFORM_PREFIX=""
if [[ "${1:-}" =~ ^(splat|SPLAT)$ ]] ; then
  TRANSFORM_PREFIX="splat_"
  shift
fi

set -x

rm -rf "${WORKING_DIR}/configurations/punet"
rm -rf "${WORKING_DIR}/sources/punet"
rm -rf "${WORKING_DIR}/binaries/punet"
rm -rf "${WORKING_DIR}/benchmarks/punet"

# Compile to flow, if scheduled_unet_flow.mlir does not exist
if [ ! -f ${PWD}/tmp/punet_flow.mlir ]; then
  echo "Compiling to flow...\n"
  "${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
    "${SCRIPT_DIR}/specs/${TRANSFORM_PREFIX}attention_and_matmul_spec.mlir" \
    "${SCRIPT_DIR}/base_ir/punet_07_18.mlir" \
    --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/punet" \
    --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/punet" \
    --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/punet" \
    --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/benchmarks/punet" \
    --compile-to=flow \
    -o "${WORKING_DIR}/tmp/punet_flow.mlir" \
    "$@"
else
  echo "Found ${PWD}/tmp/punet_flow.mlir. Adding tk kernels ..."
fi

# Insert tk kernels
python3 add_tk_kernels.py

"${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/specs/${TRANSFORM_PREFIX}attention_and_matmul_spec.mlir" \
  "${SCRIPT_DIR}/tmp/punet_tk.mlir" \
  --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/punet" \
  --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/punet" \
  --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/punet" \
  --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/benchmarks/punet" \
  --iree-scheduling-dump-statistics-file="${WORKING_DIR}/tmp/punet_scheduling_stats.txt" \
  --iree-scheduling-dump-statistics-format=csv \
  --compile-from=flow \
  -o "${WORKING_DIR}/tmp/punet.vmfb" \
  "$@"

  #--iree-hal-benchmark-dispatch-repeat-count=20 \
  #--iree-hal-executable-debug-level=3 \
  #--iree-vulkan-target-triple=rdna3-unknown-linux \
  #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
  #--iree-hal-cuda-llvm-target-arch=sm_80 \
  #--mlir-disable-threading \
