#!/bin/bash

set -xe

script_dir=$(dirname $(realpath $0))
event_name=$1
shift

echo $@
echo $script_dir

real_iree=$(command -v iree-benchmark-module)

iree-benchmark-module () {
  sudo LD_LIBRARY_PATH=/usr/local/lib runTracer.sh $real_iree $@
  python3 /usr/local/bin/rpd2tracing.py --format object trace.rpd trace.json
  python3 $script_dir/corellator.py $event_name trace.json
}

source $@
