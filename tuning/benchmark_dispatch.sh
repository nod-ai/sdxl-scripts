#! /usr/bin/env bash

set -euo pipefail

readonly INPUT="$(realpath "$1")"
readonly DIR="$(dirname "$INPUT")"
readonly DEVICE="$2"
shift 2

mkdir -p "${DIR}/failed"
readonly NAME="$(basename "$INPUT" .mlir)"

printf "Benchmarking ${INPUT} on ${DEVICE}\n"

timeout 5s ./tools/iree-benchmark-module --device="rocm://${DEVICE}" --module="${INPUT}" \
  --batch_size=1000 --benchmark_repetitions=3 > "benchmark_log_${DEVICE}.out" 2>&1 || (mv "$INPUT" "$DIR/failed" && exit 1)

MEAN_TIME="$(grep real_time_mean "benchmark_log_${DEVICE}.out" | awk '{print $2}')"
printf "%s\tMean Time: %.1f\n" "$(basename "$INPUT" .vmfb)" "$MEAN_TIME"
