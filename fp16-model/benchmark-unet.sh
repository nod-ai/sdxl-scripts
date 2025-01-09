#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -xeu

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<ipra-path-prefix>]"
  exit 1
fi

IRPA_PATH_PREFIX="${2:-/data/shark}"

# used as a workaround for lengthy initialization
# export ROCR_VISIBLE_DEVICES=0

iree-benchmark-module \
  --device=hip://$1 \
  --device_allocator=caching \
  --module=$PWD/tmp/unet.vmfb \
  --parameters=model=${IRPA_PATH_PREFIX}/scheduled_unet.irpa \
  --function=run_forward \
  --input=@sample_inputs/unet_npys/arg0_latent_model_input.npy \
  --input=@sample_inputs/unet_npys/arg1_guidanc_scale.npy \
  --input=@sample_inputs/unet_npys/arg2_prompt_embeds.npy \
  --input=@sample_inputs/unet_npys/arg3_add_text_embeds.npy \
  --input=@sample_inputs/unet_npys/arg4_add_time_ids.npy \
  --input=@sample_inputs/unet_npys/arg5_t.npy \
  --benchmark_repetitions=3

  # --device=hip://$1 \
  # --parameters=model=${IRPA_PATH_PREFIX}/scheduled_unet.irpa \
