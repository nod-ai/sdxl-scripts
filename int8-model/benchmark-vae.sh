#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-vae.sh N

set -xeu

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <batch-size> [<ipra-path-prefix>]"
  exit 1
fi

DEVICE=$1;     shift
BATCH_SIZE=$1; shift
IRPA_PATH_PREFIX="${1:-/data/amd-shark}"

re='^[0-9]+$'
if ! [[ $BATCH_SIZE =~ $re ]] ; then
   echo "error: <batch-size> must be a number"
   exit 1
fi

iree-benchmark-module \
  --device=hip://${DEVICE} \
  --hip_use_streams=true \
  --hip_allow_inline_execution=true \
  --device_allocator=caching \
  --module=$PWD/tmp/vae_decode.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/vae_decode_fp16.irpa \
  --function=decode \
  --input=${BATCH_SIZE}x4x128x128xf16 \
  --benchmark_repetitions=3
