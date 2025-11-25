#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./collect_numbers.sh

set -xeu

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

echo "====== E2E (Real Weights) ======"

echo "==> [Default]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img.sh gfx942
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

sleep 15

echo "==> [Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 default
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

sleep 15

echo "==> [Winograd]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-winograd.sh gfx942
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

sleep 15

echo "==> [Winograd-Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 winograd
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

sleep 15

echo "==> [Misa]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-misa.sh gfx942
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

sleep 15

echo "==> [Misa-Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 misa
$SCRIPT_DIR/benchmark-txt2img.sh 5 /data/amd-shark/

echo "====== E2E (Splat Weights) ======"

sleep 15

echo "==> [Default]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img.sh gfx942 splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat

sleep 10

echo "==> [Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 default splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat

sleep 10

echo "==> [Winograd]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-winograd.sh gfx942 splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat

sleep 10

echo "==> [Winograd-Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 winograd splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat

sleep 10

echo "==> [Misa]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-misa.sh gfx942 splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat

sleep 10

echo "==> [Misa-Tk]"
rm -f $SCRIPT_DIR/tmp/*.mlir $SCRIPT_DIR/tmp/*.vmfb
$SCRIPT_DIR/compile-txt2img-tk.sh gfx942 misa splat
$SCRIPT_DIR/benchmark-txt2img.sh 5 $SCRIPT_DIR/splat
