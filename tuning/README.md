# Matmul auto-tuning scripts

## Overall flow

1. Simlink all scripts and mlir/irpa files in your build dir.
2. Compile unet and dump all executable files with
   `--iree-hal-dump-executable-files-to=dump-unet`.
3. Find the matmul to tune and copy the _benchmark.mlir file to the build dir.
4. Run the tuner `./tuner.py mmt.mlir -o candidates -l 8192`.
5. Compile all candidates: `parallel ./compile_candidate.sh {} ::: candidate/*.mlir`.
6. Benchmark all candidates on GPUs 2-7:

   ```shell
   time parallel -j6 './benchmark_dispatch.sh {} $(({%}+1))' ::: candidates/compiled/*.vmfb | tee tuned.log
   ```

7. Check the winners:

   ```shell
   cat tuned.log | awk '{printf("%s, %s\n", $NF,$2);}' | sort -n | head -n5`
   ```

8. Copy the transforms script from the correspondig `.mlir` file into the TD
   spec.

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
