#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-txt2img-winograd.sh [extra flags]

set -xeu

readonly TARGET="$1"
readonly TRANSFORM_PREFIX="${2:-}"

"$PWD"/compile-clip.sh "$TARGET"

if [ "$TRANSFORM_PREFIX" == "" ] ; then
  "$PWD"/compile-scheduled-unet-winograd.sh "$TARGET"
else
  "$PWD"/compile-scheduled-unet-winograd.sh "$TARGET" "$TRANSFORM_PREFIX"
fi

"$PWD"/compile-vae.sh "$TARGET"

iree-compile "$PWD"/base_ir/sdxl_pipeline_bench_f16.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip="$TARGET" \
    --iree-rocm-bc-dir="$PWD"/../bitcode-2024-03-07 \
    --iree-global-opt-propagate-transposes=true \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-rocm-waves-per-eu=2 \
    --iree-opt-outer-dim-concat=true \
    --iree-llvmgpu-enable-prefetch \
    -o "$PWD"/tmp/full_pipeline.vmfb
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \z
