#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-scheduled-unet-misa.sh [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
shift

TRANSFORM_PREFIX=""
if [[ "${1:-}" =~ ^(splat|SPLAT)$ ]] ; then
  TRANSFORM_PREFIX="splat_"
  shift
fi

set -x

"${SCRIPT_DIR}/compile-unet-base.sh" "$IREE_COMPILE" "$CHIP" misa \
  "${SCRIPT_DIR}/specs/${TRANSFORM_PREFIX}attention_and_matmul_spec.mlir" \
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_PNDM_64_1024x1024_fp16_unet_30.mlir" \
  --iree-hal-dump-executable-configurations-to=configurations/scheduled_unet_misa \
  --iree-hal-dump-executable-sources-to=sources/scheduled_unet_misa \
  --iree-hal-dump-executable-binaries-to=binaries/scheduled_unet_misa \
  --iree-hal-dump-executable-benchmarks-to=benchmarks/scheduled_unet_misa \
  -o "${PWD}/tmp/scheduled_unet.vmfb" \
  "$@"
