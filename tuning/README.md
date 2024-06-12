# Matmul auto-tuning scripts

## Overall flow

1. Simlink all scripts and mlir/irpa files in your build dir.
   - Symlink `iree-build-dir/tools` inside `sdxl-scripts/tuning`.
   - Symlink UNet MLIR and weights based on `unet.sh`.
     - The full UNet is in [https://github.com/nod-ai/sdxl-scripts/tree/main/base_ir](https://github.com/nod-ai/sdxl-scripts/tree/main/base_ir).
     - Check: `stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir`.
     - The weights are on the machine under `/data`.
     - Usage: `/data/home/perf/data/shark/scheduled_unet.irpa`.

2. Copy the attention/matmul spec as `config.mlir` in the tuning dir. 
```shell
cd tuning
cp ../specs/attention_and_matmul_spec.mlir config.mlir
```

3. Temporarily comment out all the existing configs in `config.mlir`.
   - Example:
     ```mlir
     // , @match_mmt_2048x10240x1280 -> @apply_op_config
     // , @match_mmt_2048x1280x5120 -> @apply_op_config
     // , @match_mmt_2048x1280x1280 -> @apply_op_config
     ```

4. Compile a baseline unet
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

8. Copy the lines in `<id>_config.mlir` file from the corresponding candidate into the `config.mlir`.

9. Copy the transforms script from the correspondig .mlir file into the TD spec.

## Correctness validation

We tune on full dispatches that often contain fused `linalg.generic`s in
addition to the main matmul. For the purpose of correctness validation, we
extract the exact matmul shape into [mmt_unet.mlir](./mmt_net.mlir) and validate
just the matmul part. The file contains the exact instructions for benchmarking
pure matmuls.

Validation is done in a similar way -- we produce two inputs with
`gen_matrices.py`, give them to inuts with `--input=@matA.npy`,and extract as
outputs with `--output=@outC.npy`. Then we use `./evaluate.py expected.npy
outC.npy` to confirm the output matches the expectation.
