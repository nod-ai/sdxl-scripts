#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
shift

set -x

"${SCRIPT_DIR}/compile-unet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir" \
  --iree-hal-dump-executable-configurations-to=configurations/unet \
  --iree-hal-dump-executable-sources-to=sources/unet \
  --iree-hal-dump-executable-binaries-to=binaries/unet \
  --iree-hal-dump-executable-benchmarks-to=benchmarks/unet \
  -o "${PWD}/tmp/unet.vmfb" \
  "$@"
