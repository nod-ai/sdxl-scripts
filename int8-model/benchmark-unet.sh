#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <batch-size> [<ipra-path-prefix>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly HIP_DEVICE="$1"
readonly BATCH_SIZE="$2"
if ! [[ "${BATCH_SIZE}" =~ ^(1|4|8|14)$ ]]; then
  echo "Allowed batch-sizes: 1, 4, 8, 14"
  exit 1
fi
INPUT_PATH="${SCRIPT_DIR}/unet_npys"
INPUTS="--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/latent_model_input.npy \
--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/guidance_scale.npy \
--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/prompt_embeds.npy \
--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/add_text_embeds.npy \
--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/add_time_ids.npy \
--input=@${INPUT_PATH}/unet_inputs_bs${BATCH_SIZE}/t.npy"

IRPA_PATH_PREFIX="${3:-/data/shark}"

"$IREE_BENCHMARK" \
  --device="hip://${HIP_DEVICE}" \
  --device_allocator=caching \
  --module="${SCRIPT_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  --parameters=model=${IRPA_PATH_PREFIX}/sdxl_unet_int8_dataset.irpa \
  --function=main \
  $INPUTS \
  --benchmark_repetitions=3
