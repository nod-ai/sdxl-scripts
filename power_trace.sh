#!/bin/bash

set -xe

script_dir=$(dirname $(realpath $0))

real_iree=$(command -v iree-benchmark-module)

iree-benchmark-module () {
  sudo LD_LIBRARY_PATH=/usr/local/lib runTracer.sh $real_iree $@
  python3 $script_dir/corellator.py trace.rpd
}

source $@
