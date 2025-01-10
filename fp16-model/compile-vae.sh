#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-vae.sh [extra flags]

set -xeu

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

if (( $# < 1 )); then
  echo "usage: $0 <target-chip>"
  exit 1
fi
CHIP=$1
shift

iree-compile "${SCRIPT_DIR}/base_ir//stable_diffusion_xl_base_1_0_bs1_960x1024_fp16_vae_decomp_attn.mlir" \
	     --iree-hal-target-backends=rocm \
	     --iree-hip-target=${CHIP} \
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
	     --iree-llvmgpu-enable-prefetch=true \
	     --iree-dispatch-creation-enable-aggressive-fusion \
	     --iree-codegen-llvmgpu-use-vector-distribution=true \
	     --iree-hal-dump-executable-files-to=tmp/dump \
	     -o $PWD/tmp/vae_decode.vmfb \
	     "$@"

