#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-scheduled-unet.sh <target-chip> <default|winograd|misa|hybrid> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
readonly MODE="$2"
shift 2

TRANSFORM_PREFIX=""
if [[ "${1:-}" =~ ^(splat|SPLAT)$ ]] ; then
  TRANSFORM_PREFIX="splat_"
  shift
fi

set -x

# Compile to flow, if scheduled_unet_flow.mlir does not exist
if [ ! -f ${PWD}/tmp/scheduled_unet_flow.mlir ]; then
  echo "Compiling to flow...\n"
  "${SCRIPT_DIR}/compile-unet-base.sh" "$IREE_COMPILE" "$CHIP" "$MODE" \
    "${SCRIPT_DIR}/specs/${TRANSFORM_PREFIX}attention_and_matmul_spec.mlir" \
    "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_PNDM_64_1024x1024_fp16_unet_30.mlir" \
    --iree-hal-dump-executable-configurations-to=configurations/scheduled_unet \
    --iree-hal-dump-executable-sources-to=sources/scheduled_unet \
    --iree-hal-dump-executable-binaries-to=binaries/scheduled_unet \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/scheduled_unet \
    --compile-to=flow \
    -o "${PWD}/tmp/scheduled_unet_flow.mlir" \
    "$@"
else
  echo "Found ${PWD}/tmp/scheduled_unet_flow.mlir. Adding tk kernels ..."
fi

# Insert tk kernels
python3 add_tk_kernels.py

# Compile from flow
"${SCRIPT_DIR}/compile-unet-base.sh" "$IREE_COMPILE" "$CHIP" "$MODE" \
  "${SCRIPT_DIR}/specs/${TRANSFORM_PREFIX}attention_and_matmul_spec.mlir" \
  "${PWD}/tmp/scheduled_unet_tk.mlir" \
  --iree-hal-dump-executable-configurations-to=configurations/scheduled_unet \
  --iree-hal-dump-executable-sources-to=sources/scheduled_unet \
  --iree-hal-dump-executable-binaries-to=binaries/scheduled_unet \
  --iree-hal-dump-executable-benchmarks-to=benchmarks/scheduled_unet \
  --compile-from=flow \
  -o "${PWD}/tmp/scheduled_unet.vmfb" \
  "$@"

  #--iree-hal-benchmark-dispatch-repeat-count=20 \
  #--iree-hal-executable-debug-level=3 \
  #--iree-vulkan-target-triple=rdna3-unknown-linux \
  #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
  #--iree-hal-cuda-llvm-target-arch=sm_80 \
  #--mlir-disable-threading \
