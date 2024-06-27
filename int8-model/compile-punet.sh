#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}
shift

set -x

"${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/base_ir/punet_06_26_all_signed.mlir" \
  --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/punet" \
  --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/punet" \
  --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/punet" \
  --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/benchmarks/punet" \
  --iree-scheduling-dump-statistics-file="${WORKING_DIR}/tmp/punet_scheduling_stats.txt" \
  --iree-scheduling-dump-statistics-format=csv \
  -o "${WORKING_DIR}/tmp/punet.vmfb" \
  "$@"

  #--iree-hal-benchmark-dispatch-repeat-count=20 \
  #--iree-hal-executable-debug-level=3 \
  #--iree-vulkan-target-triple=rdna3-unknown-linux \
  #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
  #--iree-hal-cuda-llvm-target-arch=sm_80 \
  #--mlir-disable-threading \
