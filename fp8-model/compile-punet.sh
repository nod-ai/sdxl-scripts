#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 <hip-target-chip>"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
readonly BATCH_SIZE=1
shift 1

readonly WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}

set -x

"${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_scheduled_unet_bs${BATCH_SIZE}_64_1024x1024_fp8.mlir" \
  -o "${WORKING_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  "$@"
