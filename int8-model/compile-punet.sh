#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 <hip-target-chip> <chip-configuration-mode>"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
readonly CHIP_CONFIGURATION="$2"
EXTRA_FLAGS="${@:3}"
if ! [[ "${CHIP_CONFIGURATION}" =~ ^(cpx|qpx)$ ]]; then
  echo "Allowed chip-configuration-modes: cpx, qpx"
  exit 1
fi

WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}
shift

set -x

rm -rf "${WORKING_DIR}/configurations/punet"
rm -rf "${WORKING_DIR}/sources/punet"
rm -rf "${WORKING_DIR}/binaries/punet"
rm -rf "${WORKING_DIR}/benchmarks/punet"

"${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/specs/attention_and_matmul_spec_punet_mi300_${CHIP_CONFIGURATION}.mlir" \
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_bs1_64_1024x1024_i8_punet.mlir" \
  --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/punet" \
  --iree-hal-dump-executable-intermediates-to="${WORKING_DIR}/intermediates/punet" \
  --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/punet" \
  --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/punet" \
  --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/benchmarks/punet" \
  --iree-scheduling-dump-statistics-file="${WORKING_DIR}/tmp/punet_scheduling_stats.txt" \
  --iree-scheduling-dump-statistics-format=csv \
  -o "${WORKING_DIR}/tmp/punet.vmfb" \
  $EXTRA_FLAGS

  #--iree-hal-benchmark-dispatch-repeat-count=20 \
  #--iree-hal-executable-debug-level=3 \
  #--iree-vulkan-target-triple=rdna3-unknown-linux \
  #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
  #--iree-hal-cuda-llvm-target-arch=sm_80 \
  #--mlir-disable-threading \
