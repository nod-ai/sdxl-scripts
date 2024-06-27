#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-punet.sh <target-chip> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
readonly INPUT="${SCRIPT_DIR}/base_ir/punet_06_26_all_signed.mlir"
WORKING_DIR=${WORKING_DIR:-${SCRIPT_DIR}}
ROCM_BC_DIR=${ROCM_BC_DIR:-${SCRIPT_DIR}/../bitcode-2024-03-07}

shift

set -x

"${IREE_COMPILE}" "${INPUT}" \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target-chip="${CHIP}" \
  --iree-rocm-bc-dir="${ROCM_BC_DIR}" \
  --iree-opt-const-eval=false \
  --iree-opt-data-tiling=false \
  --iree-flow-enable-aggressive-fusion \
  --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/configurations/punet" \
  --iree-hal-dump-executable-sources-to="${WORKING_DIR}/sources/punet" \
  --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/binaries/punet" \
  --iree-scheduling-dump-statistics-file="${WORKING_DIR}/tmp/punet_scheduling_stats.txt" \
  --iree-scheduling-dump-statistics-format=csv \
  -o "${WORKING_DIR}/tmp/punet.vmfb" \
  --compile-to=executable-sources \
  "$@"
