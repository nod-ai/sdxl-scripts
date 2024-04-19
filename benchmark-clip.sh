#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-clip.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/shark}"

iree-benchmark-module \
  --device=rocm://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/prompt_encoder.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/prompt_encoder.irpa \
  --function=encode_prompts \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --input=1x64xi64 \
  --benchmark_repetitions=3
