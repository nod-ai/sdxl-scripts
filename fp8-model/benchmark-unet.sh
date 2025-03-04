#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly HIP_DEVICE="$1"
readonly BATCH_SIZE=1
readonly INPUT_PATH="${SCRIPT_DIR}/unet_npys/unet_inputs_bs${BATCH_SIZE}"

INPUTS="--input=@${INPUT_PATH}/run_forward_input_0.npy \
--input=@${INPUT_PATH}/run_forward_input_1.npy \
--input=@${INPUT_PATH}/run_forward_input_2.npy \
--input=@${INPUT_PATH}/run_forward_input_3.npy \
--input=@${INPUT_PATH}/run_forward_input_4.npy \
--input=@${INPUT_PATH}/run_forward_input_5.npy \
--input=@${INPUT_PATH}/run_forward_input_6.npy \
--input=@${INPUT_PATH}/run_forward_input_7.npy"

IRPA_PATH_PREFIX="${2:-/data/shark}"

"$IREE_BENCHMARK" \
  --device="hip://${HIP_DEVICE}" \
  --device_allocator=caching \
  --module="${SCRIPT_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  --parameters=model=${IRPA_PATH_PREFIX}/sdxl_unet_fp8_ocp_dataset.irpa \
  --function=run_forward \
  $INPUTS \
  --benchmark_repetitions=3
