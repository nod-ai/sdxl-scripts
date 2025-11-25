#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-vae.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/amd-shark}"

iree-benchmark-module \
  --device=rocm://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/vae_decode.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/vae_decode.irpa \
  --function=main \
  --input=1x4x128x128xf16 \
  --benchmark_repetitions=3
