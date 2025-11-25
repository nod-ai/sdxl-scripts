#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/amd-shark}"

iree-benchmark-module \
  --device=hip://$1 \
  --hip_use_streams=true \
  --hip_allow_inline_execution=true \
  --device_allocator=caching \
  --module=$PWD/tmp/punet.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/sdxl_unet_int8_dataset.irpa \
  --function=main \
  --input=1x4x128x128xf16 \
  --input=1xsi32 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --benchmark_repetitions=3
