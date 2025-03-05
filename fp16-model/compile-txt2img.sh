#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-txt2img.sh [extra flags]

set -xeu

readonly TARGET="$1"
readonly TRANSFORM_PREFIX="${2:-}"

"$PWD"/compile-clip.sh "$TARGET"

if [ "$TRANSFORM_PREFIX" == "" ] ; then
  "$PWD"/compile-scheduled-unet.sh "$TARGET"
else
  "$PWD"/compile-scheduled-unet.sh "$TARGET" "$TRANSFORM_PREFIX"
fi

"$PWD"/compile-vae.sh "$TARGET"

iree-compile "$PWD"/base_ir/sdxl_pipeline_bench_f16.mlir \
    --iree-hal-target-backends=rocm \
    --iree-hip-target="$TARGET" \
    --iree-hip-bc-dir="$(hipconfig --rocmpath)/amdgcn/bitcode" \
    --iree-global-opt-propagate-transposes=true \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-hip-waves-per-eu=2 \
    --iree-opt-outer-dim-concat=true \
    --iree-llvmgpu-enable-prefetch \
    -o "$PWD"/tmp/full_pipeline.vmfb
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \z
