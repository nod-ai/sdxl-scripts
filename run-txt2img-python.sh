#!/bin/bash

set -xeu

if [ $# -ne 3 ]; then
    echo "Usage: $0 <batch_count> <device_id> <weights_path> <extra_args>"
    exit 1
fi

batch_count="--batch_count=$1"
device="rocm://$2"
weights_path="--external_weights_dir=$3"

python3 ../SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_compiled_pipeline.py \
  --precision=fp16 \
  --external_weights=irpa \
  --device=rocm \
  --rt_device=$device \
  --iree_target_triple=gfx942 \
  --scheduler_id=PNDM \
  --num_inference_steps=30 \
  --pipeline_dir=./tmp/ \
  $weights_path \
  --attn_spec=default  \
  $batch_count \
  $@ | tee output.txt

grep -oE '[^/]+\.png' output.txt > filenames.txt

readarray -t filenames < filenames.txt

for filename in $filenames; do
    mv "$filename" /gen_imgs
done
