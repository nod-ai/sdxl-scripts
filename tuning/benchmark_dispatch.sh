#! /usr/bin/env bash

set -euo pipefail

readonly INPUT="$(realpath "$1")"
readonly DIR="$(dirname "$INPUT")"
readonly DEVICE="$2"
shift 2

mkdir -p "${DIR}/benchmark_failed"
readonly NAME="$(basename "$INPUT" .mlir)"

# printf "Benchmarking $(basename ${INPUT}) on ${DEVICE}\n"

timeout 20s ./tools/iree-benchmark-module --device="rocm://${DEVICE}" --module="${INPUT}" \
  --batch_size=1000 --benchmark_repetitions=3 > "${DIR}/benchmark_log_${DEVICE}.out" 2>&1 || (mv "$INPUT" "${DIR}/benchmark_failed" && exit 0)

MEAN_TIME="$(grep --text real_time_mean "${DIR}/benchmark_log_${DEVICE}.out" | awk '{print $2}')"
printf "%s\tMean Time: %.1f\n" "$(basename "$INPUT" .vmfb)" "$MEAN_TIME"
