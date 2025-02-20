#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-vae.sh N
# IRPA file can be found at https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl/weights/stable_diffusion_xl_base_1_0_vae_dataset_fp16.irpa

set -xeu

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <batch-size> [<ipra-path-prefix>]"
  exit 1
fi
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

DEVICE=$1;     shift
BATCH_SIZE=$1; shift
IRPA_PATH_PREFIX="${1:-/data/shark}"

re='^[0-9]+$'
if ! [[ $BATCH_SIZE =~ $re ]] ; then
   echo "error: <batch-size> must be a number"
   exit 1
fi

INPUT_PATH="${SCRIPT_DIR}/vae_npys"

iree-benchmark-module \
  --device=hip://${DEVICE} \
  --device_allocator=caching \
  --module=$PWD/tmp/vae_decode.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/stable_diffusion_xl_base_1_0_vae_dataset_fp16.irpa \
  --function=decode \
  --input=@${INPUT_PATH}/vae_inputs_bs${BATCH_SIZE}/latents.npy \
  --benchmark_repetitions=3
