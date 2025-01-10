#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-vae.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

readonly IREE_BENCHMARK_MODULE="$(which iree-benchmark-module)"
readonly HIP_DEVICE_ID="$1"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IRPA_PATH_PREFIX="${2:-/data/shark}"

"$IREE_BENCHMARK_MODULE" \
  --device=hip://$HIP_DEVICE_ID \
  --device_allocator=caching \
  --module="${SCRIPT_DIR}/tmp/vae_decode.vmfb" \
  --parameters=model="${IRPA_PATH_PREFIX}/vae_decode_fp16.irpa" \
  --function=decode \
  --input="@${SCRIPT_DIR}/sample_inputs/vae_npys/random_vae_inputs.npy" \
  --benchmark_repetitions=3

  # --parameters=model=${IRPA_PATH_PREFIX}/stable_diffusion_xl_base_1_0_vae_fp16.safetensors \
  # --input=1x4x120x128xf16 \
