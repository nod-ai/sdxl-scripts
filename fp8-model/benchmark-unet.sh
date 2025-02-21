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
if ! [[ "${BATCH_SIZE}" =~ ^(4|8|16|18)$ ]]; then
  echo "Allowed batch-sizes: 4, 8, 16, 18"
  exit 1
fi
INPUT_PATH="${SCRIPT_DIR}/unet_npys/unet_inputs_bs${BATCH_SIZE}"

INPUTS="--input=@${INPUT_PATH}/run_forward_input_0.npy \
--input=@${INPUT_PATH}/run_forward_input_1.npy \
--input=@${INPUT_PATH}/run_forward_input_2.npy \
--input=@${INPUT_PATH}/run_forward_input_3.npy \
--input=@${INPUT_PATH}/run_forward_input_4.npy \
--input=@${INPUT_PATH}/run_forward_input_5.npy \
--input=@${INPUT_PATH}/run_forward_input_6.npy \
--input=@${INPUT_PATH}/run_forward_input_7.npy"

IRPA_PATH_PREFIX="${3:-/data/shark}"

"$IREE_BENCHMARK" \
  --device="hip://${HIP_DEVICE}" \
  --device_allocator=caching \
  --module="${SCRIPT_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  --parameters=model=${IRPA_PATH_PREFIX}/punet_fp8_weights.irpa \
  --function=run_forward \
  $INPUTS \
  --benchmark_repetitions=3
