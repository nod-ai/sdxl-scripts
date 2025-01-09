#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

# if (( $# != 1 && $# != 2 )); then
#   echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
#   exit 1
# fi

# IRPA_PATH_PREFIX="${2:-/data/shark}"

# used as a workaround for lengthy initialization
# export ROCR_VISIBLE_DEVICES=0

iree-benchmark-module \
  --device=hip \
  --device_allocator=caching \
  --module=$PWD/tmp/unet.vmfb \
  --parameters=model=/home/nmeganat/SHARK-ModelDev/weights/stable_diffusion_xl_base_1_0_unet_fp16.safetensors \
  --function=run_forward \
  --input=1x4x120x128xf16 \
  --input=1xf16 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --benchmark_repetitions=3

  # --device=hip://$1 \
  # --parameters=model=${IRPA_PATH_PREFIX}/scheduled_unet.irpa \