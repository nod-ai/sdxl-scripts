#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

# IRPA file: https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl-scripts-weights/scheduled_unet_fp16.irpa
# Size: 5135167488
# md5sum: 4ca340fcea6533e0693bca895991f12c
IRPA_PATH_PREFIX="${2:-/data/shark}"

iree-benchmark-module \
  --device=hip://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/unet.vmfb \
  --parameters=model="${IRPA_PATH_PREFIX}/scheduled_unet_fp16.irpa" \
  --function=main \
  --input=1x4x128x128xf16 \
  --input=1xi64 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --benchmark_repetitions=3
