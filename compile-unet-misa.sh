#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh [extra flags]

set -xeu

if (( $# != 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi

iree-compile $PWD/base_ir/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=$1 \
    --iree-rocm-bc-dir=$PWD/bitcode-2024-03-07 \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-data-tiling=false \
    --iree-opt-const-eval=false \
    --iree-rocm-waves-per-eu=2 \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-flow-enable-aggressive-fusion \
    --iree-global-opt-enable-fuse-horizontal-contractions=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-execution-model=async-external \
    --iree-codegen-transform-dialect-library=$PWD/specs/attention_and_matmul_spec.mlir \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))" \
    --iree-hal-dump-executable-configurations-to=configurations/unet-misa \
    --iree-hal-dump-executable-sources-to=sources/unet-misa \
    --iree-hal-dump-executable-binaries-to=binaries/unet-misa \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/unet-misa \
    --iree-preprocessing-transform-spec-filename=$PWD/specs/misa_unet_spec.mlir \
    -o $PWD/tmp/unet_misa.vmfb

    # --iree-flow-dump-dispatch-graph \
    # --iree-flow-break-dispatch=@forward:igemm_fwd_gtcx3_nhwc_fp16_bx0_ex1_bt256x128x32_wt32x32x8_ws2x1_wr2x2_ta1x8x4x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs \
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \
