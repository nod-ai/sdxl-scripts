#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-txt2img.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/shark}"

echo "Benchmarking SDXL pipeline..."
iree-benchmark-module \
 --device=rocm://$1 \
 --device_allocator=caching \
 --module=$PWD/tmp/prompt_encoder.vmfb \
 --parameters=model=${IRPA_PATH_PREFIX}/prompt_encoder.irpa \
 --module=$PWD/tmp/scheduled_unet.vmfb \
 --parameters=model=${IRPA_PATH_PREFIX}/scheduled_unet.irpa \
 --module=$PWD/tmp/vae_decode.vmfb \
 --parameters=model=${IRPA_PATH_PREFIX}/vae_decode.irpa \
 --module=$PWD/tmp/full_pipeline.vmfb \
 --function=tokens_to_image \
 --input=1x4x128x128xf16 \
 --input=1xf16 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --benchmark_repetitions=3

# echo "Benchmarking CLIP..."
# $PWD/benchmark-clip.sh $1 $2
# echo "Benchmarking VAE..."
# $PWD/benchmark-vae.sh $1 $2
# echo "Benchmarking UNet..."
# $PWD/benchmark-scheduled-unet.sh $1 $2
