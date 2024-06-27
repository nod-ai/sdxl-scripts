#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
shift

set -x

"${SCRIPT_DIR}/compile-unet-base.sh" "$IREE_COMPILE" "$CHIP" \
  "${SCRIPT_DIR}/base_ir/punet_06_26_all_signed.mlir" \
  --iree-hal-dump-executable-configurations-to=configurations/unet \
  --iree-hal-dump-executable-sources-to=sources/unet \
  --iree-hal-dump-executable-binaries-to=binaries/unet \
  --iree-hal-dump-executable-benchmarks-to=benchmarks/unet \
  -o "${PWD}/tmp/unet.vmfb" \
  "$@"

  #--iree-hal-benchmark-dispatch-repeat-count=20 \
  #--iree-hal-executable-debug-level=3 \
  #--iree-vulkan-target-triple=rdna3-unknown-linux \
  #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
  #--iree-hal-cuda-llvm-target-arch=sm_80 \
  #--mlir-disable-threading \
