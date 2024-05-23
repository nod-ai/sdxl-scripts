#! /usr/bin/env bash

set -xeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly MODE="$1"
readonly INPUT="$(realpath "$2")"
shift 2

"${SCRIPT_DIR}/../compile-unet-base.sh" "${SCRIPT_DIR}/tools/iree-compile" gfx942 "$MODE" \
  "${SCRIPT_DIR}/config.mlir" \
  "$INPUT" \
  "$@"

# --iree-hal-dump-executable-files-to=dump-unet \
