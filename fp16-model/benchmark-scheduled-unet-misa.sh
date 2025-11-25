#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

IRPA_PATH_PREFIX="${2:-/data/amd-shark}"

iree-benchmark-module \
  --device=rocm://$1 \
  --device_allocator=caching \
  --module="$PWD"/tmp/scheduled_unet_misa.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/scheduled_unet.irpa \
  --function=run_forward \
  --input=1x4x128x128xf16 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --input=1xi64 \
  --benchmark_repetitions=3
