#! /usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly MODE="$1"  # Unused for punet.
readonly INPUT="$(realpath "$2")"
shift 2

readonly BASE_DIR="$(dirname "${INPUT}")"
readonly CANDIDATE="$(basename "${INPUT}" _spec.mlir)"

timeout 220s "${SCRIPT_DIR}/../int8-model/compile-punet-base.sh" "${SCRIPT_DIR}/tools/iree-compile" gfx942 \
  "$INPUT" \
  "${SCRIPT_DIR}/punet.mlir" \
  --mlir-disable-threading \
  -o "${BASE_DIR}/../../unet_candidate_${CANDIDATE}.vmfb" 2>/dev/null || echo "Input: ${INPUT} failed to compile"
