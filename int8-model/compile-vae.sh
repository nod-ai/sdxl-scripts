#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-vae.sh [extra flags]

set -xeu

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

if (( $# != 2 )); then
  echo "usage: $0 <target-chip> <batch-size>"
  exit 1
fi

CHIP=$1;       shift
BATCH_SIZE=$1; shift

readonly PREPROCESSING_FLAGS=(
"--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics), util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
)
declare -a FLAGS=("${PREPROCESSING_FLAGS[*]}")

iree-compile ${SCRIPT_DIR}/base_ir/vae_decomp_attn_bs${BATCH_SIZE}.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=${CHIP} \
    --iree-rocm-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-opt-const-eval=false \
    --iree-opt-data-tiling=false \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-flow-enable-fuse-horizontal-contractions \
    --iree-flow-enable-aggressive-fusion \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-execution-model=async-external \
    --iree-codegen-transform-dialect-library="${SCRIPT_DIR}/specs/attention_and_matmul_spec.mlir" \
    --iree-hal-dump-executable-configurations-to=configurations/vae \
    --iree-hal-dump-executable-sources-to=sources/vae \
    --iree-hal-dump-executable-binaries-to=binaries/vae \
    --iree-hal-dump-executable-benchmarks-to=benchmarks/vae \
    -o $PWD/tmp/vae_decode.vmfb \
    "${FLAGS[@]}" \
    "$@"
    #--iree-opt-outer-dim-concat=true \
    #--iree-rocm-waves-per-eu=2 \
