#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <default|winograd|misa> <attention_matmul_spec_file> <input mlir> -o <output vmfb> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi

readonly CHIP="$2"

readonly MODE="$3"
USE_WINOGRAD=0
USE_MISA=0
if [[ $MODE =~ "winograd" ]] ; then
  USE_WINOGRAD=1
elif [[ $MODE =~ "misa" ]] ; then
  USE_MISA=1
fi

readonly ATTENTION_SPEC="$(realpath "$4")"
if [ ! -f "$ATTENTION_SPEC" ] ; then
  echo "Specified attention spec file not found: ${ATTENTION_SPEC}"
  exit 1
fi

readonly INPUT="$(realpath "$5")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

readonly SPEC_DIR="$(realpath "$SCRIPT_DIR"/specs)"
if [ ! -d "$SPEC_DIR" ] ; then
  echo "Spec directory not found: ${SPEC_DIR}"
  exit 1
fi

shift 5

readonly DEFAULT_FLAGS=(
  "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))"
)

readonly WINOGRAD_PIPELINE=("builtin.module("
  "iree-preprocessing-transform-interpreter{transform-spec-path=${SPEC_DIR}/winograd_conv_spec.mlir},"
  "util.func(iree-linalg-ext-convert-conv2d-to-winograd),"
  "iree-preprocessing-transpose-convolution-pipeline,"
  "util.func(iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}))"
)

readonly WINOGRAD_FLAGS=(
  "--iree-opt-const-expr-max-size-increase-threshold=1000000000000000"
  "--iree-preprocessing-pass-pipeline=${WINOGRAD_PIPELINE[*]}"
)

readonly MISA_FLAGS=(
  "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))"
  "--iree-hal-executable-object-search-path=${SPEC_DIR}"
  "--iree-preprocessing-transform-spec-filename=${SPEC_DIR}/misa_unet_spec.mlir"
)

declare -a FLAGS=("${DEFAULT_FLAGS[*]}")
if [ "$USE_WINOGRAD" = 1 ] ; then
  FLAGS=("${WINOGRAD_FLAGS[@]}")
elif [ "$USE_MISA" = 1 ] ; then
  FLAGS=("${MISA_FLAGS[@]}")
fi

set -x

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target-chip="$CHIP" \
    --iree-rocm-bc-dir="${SCRIPT_DIR}/../bitcode-2024-03-07" \
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
    --iree-codegen-transform-dialect-library="$ATTENTION_SPEC" \
    "${FLAGS[@]}" \
    "$@"
