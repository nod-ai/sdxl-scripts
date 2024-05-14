#! /usr/bin/env bash

set -xeuo pipefail

readonly INPUT="$(realpath "$1")"
shift

tools/iree-compile \
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
    --iree-codegen-transform-dialect-library="$(realpath config.mlir)" \
    "$INPUT" "$@"

# --iree-hal-dump-executable-files-to=dump-unet \
