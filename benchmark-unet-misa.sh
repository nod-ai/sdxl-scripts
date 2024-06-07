#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

# if (( $# != 1 && $# != 2 )); then
#   echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
if (($# != 1)); then
  echo "usage: $0 <hip-device-id>"
  exit 1
fi

iree-benchmark-module \
  --device=rocm://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/unet_misa.vmfb \
  --parameters=model=$PWD/unet.irpa \
  --function=main \
  --input=1x4x128x128xf16 \
  --input=1xi64 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --benchmark_repetitions=3
