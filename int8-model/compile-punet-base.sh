#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <attention_matmul_spec_file> <input mlir> -o <output vmfb> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi

readonly CHIP="$2"

readonly ATTENTION_SPEC="$(realpath "$3")"
if [ ! -f "$ATTENTION_SPEC" ] ; then
  echo "Specified attention spec file not found: ${ATTENTION_SPEC}"
  exit 1
fi

readonly INPUT="$(realpath "$4")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi


shift 4

# readonly DEFAULT_FLAGS=(
# "--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental), iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})"
# )
readonly DEFAULT_FLAGS=(
"--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
)
declare -a FLAGS=("${DEFAULT_FLAGS[*]}")

set -x

"$IREE_COMPILE" "$INPUT" \
    --iree-config-add-tuner-attributes \
    --iree-hal-target-backends=rocm \
    --iree-hip-target="$CHIP" \
    --iree-hip-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-hal-indirect-command-buffers=true \
    --iree-hal-memoization=true \
    --iree-stream-resource-memory-model=discrete \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-strip-assertions \
    --iree-dispatch-creation-enable-aggressive-fusion \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=true \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=false \
    --iree-codegen-transform-dialect-library="$ATTENTION_SPEC" \
    "${FLAGS[@]}" \
    "$@"
