#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-vae.sh [extra flags]

set -xeu

if (( $# != 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"

readonly SPEC_DIR="$(realpath "$SCRIPT_DIR"/specs)"
if [ ! -d "$SPEC_DIR" ] ; then
  echo "Spec directory not found: ${SPEC_DIR}"
  exit 1
fi

readonly CHIP="$1"

readonly PREPROCESSING=(
  "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)"
)

"$IREE_COMPILE" $SCRIPT_DIR/base_ir/stable_diffusion_xl_base_1_0_bs1_960x1024_fp16_vae_decomp_attn.mlir \
    --iree-hal-target-backends=rocm \
    --iree-hip-target="$CHIP" \
    --iree-hip-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-hal-indirect-command-buffers=true \
    --iree-stream-resource-memory-model=discrete \
    --iree-hip-legacy-sync=false --iree-hal-memoization=true \
    --iree-opt-strip-assertions \
    --iree-opt-outer-dim-concat=true \
    --iree-hip-waves-per-eu=2 \
    --iree-llvmgpu-enable-prefetch \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-dispatch-creation-enable-aggressive-fusion=true \
    --iree-codegen-llvmgpu-test-tile-and-fuse-matmul=true \
    -o=$SCRIPT_DIR/tmp/vae_decode.vmfb \
    "${PREPROCESSING[@]}"
