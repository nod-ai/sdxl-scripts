#! /usr/bin/env bash

set -euo pipefail

readonly INPUT="$(realpath "$1")"
shift

readonly BASE_DIR="$(dirname "${INPUT}")"
readonly CANDIDATE="$(basename "${INPUT}" _spec.mlir)"

timeout 240s tools/iree-compile unet.mlir \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip=gfx942 \
    --iree-rocm-bc-dir=bitcode \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-data-tiling=false \
    --iree-opt-const-eval=false \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-rocm-waves-per-eu=2 \
    --iree-flow-enable-aggressive-fusion \
    --iree-global-opt-enable-fuse-horizontal-contractions=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-execution-model=async-external \
    --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))" \
    --iree-codegen-transform-dialect-library="${INPUT}" \
    --mlir-disable-threading \
    -o "${BASE_DIR}/../../unet_candidate_${CANDIDATE}.vmfb" || echo "Input: ${INPUT} failed to compile"
