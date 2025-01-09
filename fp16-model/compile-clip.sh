#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-clip.sh [extra flags]

set -xeu

if (( $# != 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi

iree-compile $PWD/base_ir/fp16-model/base_ir/stable_diffusion_xl_base_1_0_bs1_64_fp16_prompt_encoder_rocm.mlir \
    --iree-hal-target-backends=rocm \
    --iree-input-type=torch \
    --iree-hip-target=$1 \
    --iree-hip-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-vm-bytecode-module-output-format=flatbuffer-binary \
    --iree-dispatch-creation-enable-aggressive-fusion \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=false \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-hip-waves-per-eu=2 \
    --iree-codegen-llvmgpu-use-vector-distribution=true \
    --iree-llvmgpu-enable-prefetch=true \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))" \
    --iree-hal-dump-executable-configurations-to=configurations/clip \
    --iree-hal-dump-executable-sources-to=sources/clip \
    --iree-hal-dump-executable-binaries-to=binaries/clip \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/clip \
    -o $PWD/tmp/prompt_encoder.vmfb

    #--iree-codegen-transform-dialect-library=$PWD/specs/attention_and_matmul_spec.mlir \
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \
