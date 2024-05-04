#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh [extra flags]

set -xeu

if (( $# != 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi

iree-compile $PWD/base_ir/stable_diffusion_xl_base_1_0_PNDM_64_1024x1024_fp16_unet_30.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=$1 \
    --iree-rocm-bc-dir=$PWD/bitcode-2024-03-07 \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-const-eval=false \
    --iree-opt-data-tiling=false \
    --iree-rocm-waves-per-eu=2 \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-flow-enable-aggressive-fusion \
    --iree-global-opt-enable-fuse-horizontal-contractions=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-execution-model=async-external \
    --iree-hal-dump-executable-configurations-to=configurations/scheduled_unet \
    --iree-hal-dump-executable-sources-to=sources/scheduled_unet \
    --iree-hal-dump-executable-binaries-to=binaries/scheduled_unet \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/scheduled_unet \
    --iree-codegen-transform-dialect-library=$PWD/specs/attention_and_matmul_spec.mlir \
    --iree-opt-const-expr-max-size-increase-threshold=1000000000000000 \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-linalg-ext-convert-conv2d-to-winograd{td-library-path=$PWD/specs/winograd_conv_spec.mlir}), iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))" \
    -o $PWD/tmp/scheduled_unet.vmfb
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \
