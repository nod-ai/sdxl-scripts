#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-vae.sh [extra flags]

set -xeu

if (( $# != 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi

iree-compile $PWD/base_ir/fp16-model/base_ir/stable_diffusion_xl_base_1_0_bs1_960x1024_fp16_vae_decomp_attn.mlir \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=$1 \
    --iree-hip-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-vm-bytecode-module-output-format=flatbuffer-binary \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=false \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-opt-data-tiling=false \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" \
    --iree-codegen-transform-dialect-library=./vmfbs/attention_and_matmul_spec_mfma.mlir \
    --iree-llvmgpu-enable-prefetch=true \
    --iree-dispatch-creation-enable-aggressive-fusion \   
    --iree-flow-enable-aggressive-fusion \
    --iree-codegen-llvmgpu-use-vector-distribution=true \
    --iree-hal-dump-executable-configurations-to=configurations/vae \
    --iree-hal-dump-executable-sources-to=sources/vae \
    --iree-hal-dump-executable-binaries-to=binaries/vae \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/vae \
    -o $PWD/tmp/vae_decode.vmfb
    #--iree-codegen-transform-dialect-library=$PWD/specs/attention_and_matmul_spec.mlir \
    #--iree-hal-benchmark-dispatch-repeat-count=20 \
    #--iree-hal-executable-debug-level=3 \
    #--iree-vulkan-target-triple=rdna3-unknown-linux \
    #--iree-llvmcpu-target-triple=x86_64-unknown-linux \
    #--iree-hal-cuda-llvm-target-arch=sm_80 \
    #--mlir-disable-threading \
