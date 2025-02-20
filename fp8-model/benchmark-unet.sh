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
INPUT_PATH="${SCRIPT_DIR}/unet_npys/unet_inputs_bs${BATCH_SIZE}"

INPUTS="--input=@${INPUT_PATH}/arg0.npy \
--input=@${INPUT_PATH}/arg1.npy \
--input=@${INPUT_PATH}/arg2.npy \
--input=@${INPUT_PATH}/arg3.npy \
--input=@${INPUT_PATH}/arg4.npy \
--input=@${INPUT_PATH}/arg5.npy \
--input=@${INPUT_PATH}/arg6.npy \
--input=@${INPUT_PATH}/arg7.npy"

IRPA_PATH_PREFIX="${3:-/data/shark}"

"$IREE_BENCHMARK" \
  --device="hip://${HIP_DEVICE}" \
  --device_allocator=caching \
  --module="${SCRIPT_DIR}/tmp/punet_bs${BATCH_SIZE}.vmfb" \
  --parameters=model=${IRPA_PATH_PREFIX}/punet_fp8_weights.irpa \
  --function=run_forward \
  $INPUTS \
  --benchmark_repetitions=3
