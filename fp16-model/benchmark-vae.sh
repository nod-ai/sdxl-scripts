#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-vae.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/shark}"

iree-benchmark-module \
  --device=hip://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/vae_decode.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/vae_decode_fp16.irpa \
  --function=main \
  --input=1x4x120x128xf16 \
  --benchmark_repetitions=3
