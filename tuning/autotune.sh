#! /usr/bin/env bash

# Usage: ./autotune.sh <default|winograd> <input-benchmark-file.mlir> [extra flags]

set -xeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly MODE="$1"
readonly INPUT="$(realpath "$2")"
shift 2

declare -a RESERVED_GPUS=(
  # "GPU-32623464-6662-6132-6439-393336303539" # Erman
  # "GPU-32666166-3865-3732-3734-623364356137" # Erman
  "GPU-37353231-3735-3232-6131-393633373830" # Harsh
  # "GPU-39303930-3934-6363-3438-613361623536" # Erman
  # "GPU-39346632-3463-6335-3731-383262343263" # Jakub
  # "GPU-62343033-3431-6134-6536-343736623437" # Jakub
  # "GPU-64393336-6231-3033-6630-653365353764" # Jakub
  # "GPU-64656437-3233-6431-3763-303765373765" # Jakub
)

readonly RESERVED="$(IFS='|' ; echo "${RESERVED_GPUS[*]}")"
declare -a AVAILABLE_GPUS=(
  $(./tools/iree-run-module --dump_devices=rocm | grep '\--device=rocm' | grep -v "$RESERVED" | cut -f3 -d'/')
)
readonly NUM_GPUS=${#AVAILABLE_GPUS[@]}

if [ $NUM_GPUS -eq 0 ]; then
  echo "No available GPUs found"
  exit 2
fi

readonly PREFERRED_GPU="${AVAILABLE_GPUS[0]}"
echo "Available GPUs (${NUM_GPUS}): ${AVAILABLE_GPUS[*]}"
echo "Preffered GPU: ${PREFERRED_GPU}"

readonly BASE_DIR="tuning_$(date +%Y_%m_%d_%H_%M)"
mkdir -p "$BASE_DIR"

cp config_prolog.mlir config_epilog.mlir "${BASE_DIR}/"
readonly TEMPLATE="${BASE_DIR}/template.mlir"
readonly CANDIDATES="${BASE_DIR}/candidates"

cp "$INPUT" "$TEMPLATE"
./tune.py "$TEMPLATE" -o "$CANDIDATES" -l 4096 "$@"

ls -1v "$CANDIDATES"/*.mlir | grep -v _config.mlir | parallel ./compile_candidate.sh "$MODE" {} || true

ls -1v "$CANDIDATES/compiled"/*.vmfb > "$BASE_DIR/candidate_vmfbs.txt"
parallel -j"$NUM_GPUS" --link './benchmark_dispatch.sh' :::: "$BASE_DIR/candidate_vmfbs.txt" ::: "${AVAILABLE_GPUS[@]}" 2>&1 \
  | tee "${BASE_DIR}/results.log" || echo "Statuses: ${PIPESTATUS[*]}"

cat "${BASE_DIR}/results.log" | grep -v failed | \
  awk -v dir="$CANDIDATES" '{printf("%s\t%s/%s.mlir\t%s/configs/%s_spec.mlir\n", $NF, dir, $1, dir, $1);}' \
    | sort -n | head -n20 | tee "${BASE_DIR}/best.log"

cat "${BASE_DIR}/best.log" | grep -v '/0.mlir' | cut -f3 | parallel ./compile_unet_candidate.sh "$MODE" {}

echo "Preffered GPU: ${PREFERRED_GPU}"
for unet_candidate in "unet_baseline.vmfb" "${BASE_DIR}"/*.vmfb "unet_baseline.vmfb" ; do
  ./benchmark_unet_candidate.sh "${unet_candidate}" "$PREFERRED_GPU" || echo "Failed"
  sleep 10
done | tee "${BASE_DIR}/unet_results.log"
