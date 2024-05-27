#!/usr/bin/env python3

import sys
import shutil
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
import time

# Global constants, change it as needed
BATCH_SIZE = 1024
GPU_INDEX_STR = "GPU-32666166-3865-3732-3734-623364356137"

def abort(result):
    # usage exmaple: abort( run_command(...) ) for debug option
    if result.stderr:
        print(f"Command aborted, check in log file")
        sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Autotune script")
    parser.add_argument('mode', choices=['default', 'winograd'], help="Mode to run the script in")
    parser.add_argument('input_file', type=Path, help="Path to the input benchmark file (.mlir)")
    parser.add_argument('extra_dims', nargs=argparse.REMAINDER, help="Extra dimension info for winograd mode in tune.py")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output to stdout")

    return parser.parse_args()

def set_logging_env(args):
    log_dir = Path(f"tuning_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_name = f"autotune_{args.mode}_{args.input_file.stem}_{BATCH_SIZE}_{GPU_INDEX_STR}.log"
    log_file = log_dir / log_file_name
    setup_logging(log_file, args.verbose)

    logging.info(f"Mode: {args.mode}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"GPU_INDEX_STR: {GPU_INDEX_STR}")

    return log_dir

def setup_logging(log_file, verbose):
    handlers = [logging.FileHandler(log_file)]
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

def run_command(command, check=True):
    try:
        result = subprocess.run(command, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(result.stdout)
        if result.stderr:
            logging.error(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{command}' failed with error: {e.stderr}")
        if check:
            raise

def generate_candidates(args, log_dir):
    try:
        shutil.copy("config_prolog.mlir", log_dir / "config_prolog.mlir")
        shutil.copy("config_epilog.mlir", log_dir / "config_epilog.mlir")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    
    template_mlir = log_dir / "template.mlir"
    candidates_dir = log_dir / "candidates"

    shutil.copy(args.input_file, template_mlir)
    
    run_command(f"./tune.py {template_mlir} -o {candidates_dir} -l {BATCH_SIZE} {' '.join(args.extra_dims)}")

    candidates = sorted(candidates_dir.glob("*.mlir"))

    return candidates, candidates_dir

def compile_candidates(args, candidates):
    # this part seems to be slower than expected
    for candidate in candidates:
        if "_config.mlir" not in candidate.name:
            print(f"./compile_candidate.sh {args.mode} {candidate}")
            run_command(f"./compile_candidate.sh {args.mode} {candidate}", check=False)
    
    compiled_files = sorted((candidates / "compiled").glob("*.vmfb"))

    return compiled_files

def benchmark_top_candidates(log_dir, candidates_dir, compiled_files):
    benchmark_results = []
    for compiled_file in compiled_files:
        result = run_command(f'./benchmark_dispatch.sh {compiled_file} {GPU_INDEX_STR}', check=False)
        benchmark_results.append(result.stdout)

    results_log = log_dir / "results.log"
    with results_log.open('w') as log_file:
        log_file.writelines(benchmark_results)
    
    best_results = []
    with results_log.open('r') as log_file:
        for line in log_file:
            if "failed" not in line:
                parts = line.split()
                best_results.append((parts[-1], f"{candidates_dir}/{parts[0]}.mlir", f"{candidates_dir}/configs/{parts[0]}_spec.mlir"))

    best_results = sorted(best_results, key=lambda x: x[0])[:20]
    best_log = log_dir / "best.log"
    with best_log.open('w') as log_file:
        for result in best_results:
            log_file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")
    
    return best_log

def compile_unet_candidates(args, log_dir, best_log):
    with best_log.open('r') as log_file:
        for line in log_file:
            if '/0.mlir' not in line:
                parts = line.split('\t')
                run_command(f"./compile_unet_candidate.sh {args.mode} {parts[2]}")

    unet_candidates = ["unet_baseline.vmfb"] + list(log_dir.glob("*.vmfb")) + ["unet_baseline.vmfb"]

    return unet_candidates

def benchmark_unet(unet_candidates):
    for unet_candidate in unet_candidates:
        run_command(f"./benchmark_unet_candidate.sh {unet_candidate}")
        time.sleep(10)


def main():
    args = parse_arguments()
    log_dir = set_logging_env(args)

    candidates, candidates_dir = generate_candidates(args, log_dir)
    compiled_files = compile_candidates(candidates) # logs are not clean

    # didn't validate below
    best_log = benchmark_top_candidates(log_dir, candidates_dir, compiled_files)
    unet_candidates = compile_unet_candidates(args, log_dir, best_log)
    benchmark_unet(unet_candidates)

if __name__ == "__main__":
    main()
