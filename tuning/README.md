# IREE dispatch auto-tuning scripts

## Prerequisites
Using virtual environments:
```shell
cd tuning
python -m venv .venv
source .venv/bin/activate
```
Install python dependencies:
```shell
pip install -r ./tuner_requirements/requirements.txt
```
Using the IREE's Python bindings:
   - Building with CMake
     ```shell
     -DIREE_BUILD_PYTHON_BINDINGS=ON \
     -DPython3_EXECUTABLE="$(which python)"
     ```
   - Set environment
      ```shell
      source ../iree-build/.env && export PYTHONPATH
      ```
For more information, refer to the [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings)

### Overall flow

1. Symlink all scripts and mlir/irpa files in your build dir.
   - Symlink `iree-build-dir/tools` inside `sdxl-scripts/tuning`.
   - Symlink UNet MLIR and weights based on `unet.sh`.
     - The full unet is in `sdxl-scripts/*-model/base_ir`
     - The weights are in `sdxl-scripts/*-model/splat`.
   - Example:
   ```shell
   ln -s ~iree/iree-build-dir/tools ~/iree/sdxl-scripts/tools
   cd tuning
   ln -s ~/iree/sdxl-scripts/fp16-model/base_ir/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir unet.mlir
   ln -s ~/iree/sdxl-scripts/fp16-model/splat/scheduled_unet.irpa unet.irpa
   ```

2. Copy the attention/matmul spec as `config.mlir` in the tuning dir.
   - The full spec.mlir is in `sdxl-scripts/*-model/specs`
   - Example:
   ```shell
   cd tuning
   cp ~/iree/sdxl-scripts/fp16-model/specs/attention_and_matmul_spec.mlir config.mlir
   ```

4. Temporarily comment out all the existing configs in `config.mlir`.
   - Example:
     ```mlir
     // , @match_mmt_2048x10240x1280 -> @apply_op_config
     // , @match_mmt_2048x1280x5120 -> @apply_op_config
     // , @match_mmt_2048x1280x1280 -> @apply_op_config
     ```

5. Compile a baseline unet
```shell
./unet.sh winograd unet.mlir -o unet_baseline.vmfb --iree-hal-dump-executable-files-to=dump-winograd
```

5. Find the matmul to tune and copy the _benchmark.mlir file to the build dir.
```shell
cp dump-winograd/*_141_*benchmark.mlir ./141.mlir
```

6. Run the tuning script.
```shell
python autotune.py winograd 141.mlir --devices=1,3,5 --num-candidates=1024
```

7. Check the winner candidate IDs in `unet_results.log`

8. Copy the transform spec at the top of `<id>.mlir` file from the corresponding candidate into the `config.mlir` and uncomment them.

9. Add the match function to the entry point in `config.mlir`
   - Example:
     ```mlir
     @match_something -> @apply_op_config
     ```    

### Correctness validation

We tune on full dispatches that often contain fused `linalg.generic`s in
addition to the main matmul. For the purpose of correctness validation, we
extract the exact matmul shape into [mmt_unet.mlir](./mmt_net.mlir) and validate
just the matmul part. The file contains the exact instructions for benchmarking
pure matmuls.

Validation is done in a similar way -- we produce two inputs with
`gen_matrices.py`, give them to inuts with `--input=@matA.npy`,and extract as
outputs with `--output=@outC.npy`. Then we use `./evaluate.py expected.npy
outC.npy` to confirm the output matches the expectation.
