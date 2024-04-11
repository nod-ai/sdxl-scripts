#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh [extra flags]

set -xeu

iree-compile $PWD/base_ir/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=gfx942 \
    --iree-rocm-link-bc=true \
    --iree-rocm-bc-dir=$PWD/bitcode-2024-03-07 \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-llvmgpu-promote-filter=true \
    --iree-codegen-llvmgpu-use-conv-vector-distribute-pipeline \
    --iree-opt-data-tiling=false \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-log-swizzle-tile=4 \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-rocm-waves-per-eu=2 \
    --iree-codegen-llvmgpu-reduce-skinny-matmuls \
    --iree-execution-model=async-external \
    --iree-hal-dump-executable-configurations-to=configurations/unet \
    --iree-hal-dump-executable-sources-to=sources/unet \
    --iree-hal-dump-executable-binaries-to=binaries/unet \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/unet \
    --iree-opt-splat-parameter-file=tmp/splat_unet.irpa \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)" \
    --iree-codegen-transform-dialect-library=$PWD/specs/attention_and_matmul_spec.mlir \
    -o $PWD/tmp/sdxl_unet.vmfb "$@"
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \
