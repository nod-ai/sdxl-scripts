#! /usr/bin/env bash

set -xeuo pipefail

readonly INPUT="$(realpath "$1")"

readonly BASE_DIR="tuning_$(date +%Y_%m_%d_%H_%M)"
mkdir -p "$BASE_DIR"

cp config_prolog.mlir config_epilog.mlir "${BASE_DIR}/"
readonly TEMPLATE="${BASE_DIR}/template.mlir"
readonly CANDIDATES="${BASE_DIR}/candidates"

cp "$INPUT" "$TEMPLATE"
./tune.py "$TEMPLATE" -o "$CANDIDATES" -l 4096

ls -1v "$CANDIDATES"/*.mlir | grep -v _config.mlir | parallel ./compile_candidate.sh {} || true

ls -1v "$CANDIDATES/compiled"/*.vmfb | \
  parallel -j7 './benchmark_dispatch.sh {} $(({%}-1))' 2>&1 | tee "${BASE_DIR}/results.log" || true

cat "${BASE_DIR}/results.log" | grep -v failed | \
  awk -v dir="$CANDIDATES" '{printf("%s\t%s/%s.mlir\t%s/configs/%s_spec.mlir\n", $NF, dir, $1, dir, $1);}' \
    | sort -n | head -n20 | tee "${BASE_DIR}/best.log"

cat "${BASE_DIR}/best.log" | grep -v '/0.mlir' | cut -f3 | parallel ./compile_unet_candidate.sh {}

for unet_candidate in "unet_baseline.vmfb" "${BASE_DIR}"/*.vmfb "unet_baseline.vmfb" ; do
  ./benchmark_unet_candidate.sh "${unet_candidate}"
  sleep 10
done
