#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 <hip-target-chip> <tuning-chip-configuration-mode> <batch-size>"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
readonly CHIP_CONFIGURATION="$2"
readonly BATCH_SIZE="$3"
EXTRA_FLAGS="${@:4}"
if ! [[ "${CHIP_CONFIGURATION}" =~ ^(none|cpx|qpx)$ ]]; then
  echo "Allowed tuning-chip-configuration-modes: none, cpx, qpx"
  exit 1
fi
if ! [[ "${BATCH_SIZE}" =~ ^(1|4|8|14)$ ]]; then
  echo "Allowed batch-sizes: 1, 4, 8, 14"
  exit 1
fi

WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}
shift

set -x

"${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/specs/attention_and_matmul_spec_punet_mi300_${CHIP_CONFIGURATION}.mlir" \
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_scheduled_unet_bs${BATCH_SIZE}_64_1024x1024_fp8.mlir" \
  -o "${WORKING_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  $EXTRA_FLAGS

