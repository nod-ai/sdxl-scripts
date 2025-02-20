#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-vae.sh <target-chip> <tuning-chip-configuration-mode> <batch-size>

set -xeu

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

if (( $# != 3 )); then
  echo "usage: $0 <target-chip> <tuning-chip-configuration-mode> <batch-size>"
  exit 1
fi

WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}

rm -rf "${WORKING_DIR}/configurations/vae"
rm -rf "${WORKING_DIR}/sources/vae"
rm -rf "${WORKING_DIR}/binaries/vae"
rm -rf "${WORKING_DIR}/benchmarks/vae"

CHIP=$1;       shift
CHIP_CONFIGURATION=$1; shift
BATCH_SIZE=$1; shift

if ! [[ "${CHIP_CONFIGURATION}" =~ ^(none|cpx|qpx)$ ]]; then
  echo "Allowed tuning-chip-configuration-modes: none, cpx, qpx"
  exit 1
fi

ATTENTION_SPEC="${SCRIPT_DIR}/specs/attention_and_matmul_spec_vae_mi300_${CHIP_CONFIGURATION}.mlir"
if [ ! -f "$ATTENTION_SPEC" ] ; then
  echo "Specified attention spec file not found: ${ATTENTION_SPEC}"
  exit 1
fi

readonly PREPROCESSING_FLAGS=(
"--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental), iree-preprocessing-convert-conv-filter-to-channels-last{filter-layout=fhwc})"
)
declare -a FLAGS=("${PREPROCESSING_FLAGS[*]}")

iree-compile ${SCRIPT_DIR}/base_ir/20250220/stable_diffusion_xl_base_1_0_vae_bs1_1024x1024_fp16.mlir \
    --iree-config-add-tuner-attributes \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=${CHIP} \
    --iree-hip-bc-dir="${SCRIPT_DIR}/../bitcode-6.1.2" \
    --iree-vm-bytecode-module-output-format=flatbuffer-binary \
    --iree-codegen-gpu-native-math-precision=true \
    --iree-codegen-llvmgpu-early-tile-and-fuse-matmul=true \
    --iree-dispatch-creation-enable-aggressive-fusion \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=false \
    --iree-hal-indirect-command-buffers=true \
    --iree-hal-memoization=true \
    --iree-llvmgpu-enable-prefetch \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-strip-assertions \
    --iree-stream-resource-memory-model=discrete \
    --iree-codegen-transform-dialect-library=$ATTENTION_SPEC \
    --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/vae" \
    --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/vae" \
    --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/vae" \
    --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/benchmarks/vae" \
    -o $WORKING_DIR/tmp/vae_decode.vmfb \
    "${FLAGS[@]}" \
    "$@"
    #--iree-opt-outer-dim-concat=true \
    #--iree-rocm-waves-per-eu=2 \
