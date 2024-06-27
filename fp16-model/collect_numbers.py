#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess as sp
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def handle_exception(error):
    print ("[AggregatorError] Output : ", error.stdout)
    print ("[AggregatorError] Error : ", error.stderr)

def run_command(my_env, args, verbose=False):
    if verbose:
        print(f"[Aggregator] Running : {' '.join(args)}")
    return sp.run(args, cwd=SCRIPT_PATH, capture_output=True,
                  text=True, env=my_env, check=True)

def invoke_iree_compile(my_env, compile_script, compile_args = [],
                        verbose = False):
    args = [compile_script, 'gfx942'] + compile_args
    run_command(my_env, args, verbose)

def run_compiled_artifact(my_env, run_script, device,
                          data_path, run_args = [], verbose = False):
    args = [run_script, device, data_path] + run_args
    return run_command(my_env, args, verbose)

def delete_tmp_dir(my_env, verbose = False):
    args = ["rm", "-f", "tmp/*.mlir", "tmp/*.vmfb"]
    run_command(my_env, args, verbose)

def process_benchmark_log(log):
    lines= log.splitlines()
    timings = {}
    for line in lines[3:]:
        split = line.split()
        key = split[0].split('/')[-1].split('_')[-1]
        if key in ["mean", "median", "stddev"]:
            timings[key] = split[1]
    return timings

def compile_and_run(my_env, compile_script, compile_args,
                    run_script, device, data_path, run_args,
                    verbose=False):
    try:
        delete_tmp_dir(my_env, verbose)
        invoke_iree_compile(my_env, compile_script,
                            compile_args, verbose)
        output = run_compiled_artifact(my_env, run_script, device,
                                       data_path, run_args, verbose)
        timings = process_benchmark_log(output.stdout)
        return timings
    except sp.CalledProcessError as error:
        handle_exception(error)

def run_all_modes(my_env, default_compile_script, tk_compile_script,
                  winograd_compile_script, misa_compile_script, compile_args,
                  run_script, device, data_path, run_args, verbose=False):
    default_times = \
        compile_and_run(my_env, default_compile_script, compile_args,
                        run_script, device, data_path, run_args, verbose)
    print(f"Default: {default_times}")

    # TK
    tk_compile_args = ["default"] + compile_args
    tk_times = \
        compile_and_run(my_env, tk_compile_script, tk_compile_args, run_script,
                        device, data_path, run_args, verbose)
    print(f"Tk: {tk_times}")

    # Winograd
    winograd_times = \
        compile_and_run(my_env, winograd_compile_script, compile_args,
                        run_script, device, data_path, run_args, verbose)
    print(f"Winograd: {winograd_times}")

    # Winograd::Tk
    tk_winograd_compile_args = ["winograd"] + compile_args
    winograd_tk_times = \
        compile_and_run(my_env, tk_compile_script, tk_winograd_compile_args,
                        run_script, device, data_path, run_args, verbose)
    print(f"Winograd-TK: {winograd_tk_times}")

    # Misa
    misa_times = \
        compile_and_run(my_env, misa_compile_script, compile_args,
                        run_script, device, data_path, run_args, verbose)
    print(f"Misa: {misa_times}")

    # Misa::Tk
    tk_misa_compile_args = ["misa"] + compile_args
    misa_tk_times = \
        compile_and_run(my_env, tk_compile_script, tk_misa_compile_args,
                        run_script, device, data_path, run_args, verbose)
    print(f"Misa-TK: {misa_tk_times}")

def run_splat_and_real(my_env, default_compile_script, tk_compile_script,
                       winograd_compile_script, misa_compile_script, run_script,
                       args):
    print(f"## Real weights ##")
    run_all_modes(my_env, default_compile_script, tk_compile_script,
                  winograd_compile_script, misa_compile_script, [], run_script,
                  args.device, args.real_weights_path, [], args.verbose)

    print(f"## Splat weights ##")
    run_all_modes(my_env, default_compile_script, tk_compile_script,
                  winograd_compile_script, misa_compile_script, ["splat"],
                  run_script, args.device, args.splat_weights_path, [],
                  args.verbose)

def run_all_unet(my_env, script_dir, args):
    default_compile_script = \
        os.path.join(script_dir, "compile-scheduled-unet.sh")
    tk_compile_script = os.path.join(script_dir, "compile-scheduled-unet-tk.sh")
    winograd_compile_script = \
        os.path.join(script_dir, "compile-scheduled-unet-winograd.sh")
    misa_compile_script = \
        os.path.join(script_dir, "compile-scheduled-unet-misa.sh")
    run_script = os.path.join(script_dir, "benchmark-scheduled-unet.sh")

    print(f"# Unet #")
    run_splat_and_real(my_env, default_compile_script, tk_compile_script,
                       winograd_compile_script, misa_compile_script, run_script,
                       args)

def run_all_e2e(my_env, script_dir, args):
    default_compile_script = os.path.join(script_dir, "compile-txt2img.sh")
    tk_compile_script = os.path.join(script_dir,
                                          "compile-txt2img-tk.sh")
    winograd_compile_script = \
        os.path.join(script_dir, "compile-txt2img-winograd.sh")
    misa_compile_script = \
        os.path.join(script_dir, "compile-txt2img-misa.sh")
    run_script = os.path.join(script_dir, "benchmark-txt2img.sh")

    print(f"# E2E #")
    run_splat_and_real(my_env, default_compile_script, tk_compile_script,
                       winograd_compile_script, misa_compile_script,
                       run_script, args)


def main(args):
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_env = os.environ.copy()
    my_env["PATH"] = f"{args.iree_build_path}/tools:{my_env['PATH']}"

    run_all_unet(my_env, script_dir, args)
    run_all_e2e(my_env, script_dir, args)

def parse_arguments(argv):
    parser=argparse.ArgumentParser(description="SDXL measurement");

    parser.add_argument(
        "--iree-build-path",
        help="Path to IREE build directory",
        required=True
    )
    parser.add_argument(
        "--device",
        help="IREE ROCm device to use (either ordinal or UUID)",
        default="5"
    )
    parser.add_argument(
        "--real-weights-path",
        help="Path to real weights to use",
        default="/data/shark"
    )
    parser.add_argument(
        "--splat-weights-path",
        help="Path to splat weights to use",
        default="splat"
    )
    parser.add_argument(
        "--verbose",
        help = "Verbose mode",
        default=False
    )

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
