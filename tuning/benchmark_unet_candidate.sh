#! /usr/bin/env bash

set -euo pipefail

readonly INPUT="$(realpath "$1")"
shift 1

echo "Benchmarking: ${INPUT}"

tools/iree-benchmark-module \
  --device=rocm://4 \
  --device_allocator=caching \
  --module="${INPUT}" \
  --parameters=model=unet.irpa \
  --function=main \
  --input=1x4x128x128xf16 \
  --input=1xi64 \
  --input=2x64x2048xf16 \
  --input=2x1280xf16 \
  --input=2x6xf16 \
  --input=1xf16 \
  --benchmark_repetitions=5 2>&1 | grep real_time_median

echo
