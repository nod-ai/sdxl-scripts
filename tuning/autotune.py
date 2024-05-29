#!/usr/bin/env python3

import sys
import shutil
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
import time
import multiprocessing

'''
Sample Usage:

python autotune.py winograd 1286.mlir --tunepy-opt lhs-dims=bmk rhs-dims=bkn tile-dims=*mnk -v --batch-size=1024

'''

MAX_CONCURRENT_PROCESSES = multiprocessing.cpu_count()
print(f"max_concurrent_processes = {MAX_CONCURRENT_PROCESSES}\n")



# Default values for BATCH_SIZE and GPU_INDEX_STR, change it as needed
DEFAULT_BATCH_SIZE = 1024
DEFAULT_GPU_INDEX_STR = "GPU-32666166-3865-3732-3734-623364356137"

def abort(result):
    # usage exmaple: abort( run_command(...) ) for debug option
    if result.stderr:
        print(f"Command aborted, check log file")
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Autotune script")
    parser.add_argument('mode', choices=['default', 'winograd'], help="Compilation mode")
    parser.add_argument('input_file', type=Path, help="Path to the input benchmark file (.mlir)")
    # parser.add_argument('extra_flags', nargs=argparse.REMAINDER, help="Extra options passed to tune.py")
    parser.add_argument('--tunepy-opt', "-t", nargs='+', metavar='option', help="Extra options passed to tune.py (e.g., lhs-dims=bmk Xtune='--opt=val' Xtune='--opt2=val2')")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose output to stdout")

    # Optional arguments for BATCH_SIZE and GPU_INDEX_STR
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument('--gpu-index-str', type=str, default=DEFAULT_GPU_INDEX_STR, help=f"GPU index string (default: {DEFAULT_GPU_INDEX_STR})")

    return parser.parse_args()

def setup_logging(args: argparse.Namespace, log_dir: Path) -> Path:
    """Set up logging environment.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        log_dir (Path): The directory where logs are stored.

    Returns:
        Path: The path to the autotune.py log file
    """
    log_file_name = f"autotune_{args.mode}_{args.input_file.stem}.log"
    log_file = log_dir / log_file_name

    handlers = [logging.FileHandler(log_file)]
    if args.verbose:
        handlers.append(logging.StreamHandler())
        pass

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

    logging.info(f"Mode: {args.mode}")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Extra options for tune.py: {args.tunepy_opt}")
    logging.info(f"BATCH_SIZE: {args.batch_size}")
    logging.info(f"GPU_INDEX_STR: {args.gpu_index_str}")

    return log_file

def run_command(args: argparse.Namespace, command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and log the output.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        command (str): The command to run.
        check (bool, optional): Whether to check the command's exit status. Defaults to True.

    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    try:
        if args.verbose:
            print(command)
        result = subprocess.run(command, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.stdout:
            logging.info(result.stdout)
        if result.stderr:
            logging.error(result.stderr)
            print(f"[warning] bad reuslt: {command}")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{command}' failed with error: {e.stderr}")
        if check:
            raise

def generate_candidates(args: argparse.Namespace, base_dir: Path) -> tuple[list[Path], Path]:
    """Generate candidate files for tuning.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        base_dir (Path): The base directory for the tuning process.

    Returns:
        tuple[list[Path], Path]: A tuple containing a list of candidate files and the candidates directory.
    """
    try:
        shutil.copy("config_prolog.mlir", base_dir / "config_prolog.mlir")
        shutil.copy("config_epilog.mlir", base_dir / "config_epilog.mlir")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)
    
    template_mlir = base_dir / "template.mlir"
    candidates_dir = base_dir / "candidates"

    shutil.copy(args.input_file, template_mlir)
    
    formatted_args = ' '.join(['--' + opt for opt in args.tunepy_opt])
    run_command(args, f"./tune.py {template_mlir} -o {candidates_dir} -l {args.batch_size} {formatted_args}")

    candidates = sorted(candidates_dir.glob("*.mlir"))

    return candidates, candidates_dir

def compile_candidates(args: argparse.Namespace, candidates: list[Path], candidate_dir: Path) -> tuple[list[Path], Path]:
    """Compile candidate files.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        candidates (list[Path]): A list of candidate files to compile.
        candidate_dir (Path): The directory for the candidates.

    Returns:
        tuple[list[Path], Path]: A tuple containing a list of compiled files and the compiled files directory.
    """
    p_pool = multiprocessing.Pool(processes=MAX_CONCURRENT_PROCESSES)
    results = []
    for candidate in candidates:
        if "_config.mlir" not in candidate.name:
            # if args.verbose:
            #     print(f"./compile_candidate.sh {args.mode} {candidate}") 
            command = f"./compile_candidate.sh {args.mode} {candidate}"
            check=False
            # run_command(command, check)
            results.append(p_pool.apply_async(run_command, args=(args, command, check)))
            # Wait for at least one process to finish before adding more tasks
            if len(results) >= MAX_CONCURRENT_PROCESSES:
                result = results[0].get()
                results.pop(0)
    p_pool.close()

    # Wait for the remaining processes to complete
    for result in results:
        done = result.get()

    p_pool.join()

    compiled_dir = candidate_dir / "compiled"
    compiled_files = sorted(compiled_dir.glob("*.vmfb"))

    return compiled_files, compiled_dir

def benchmark_top_candidates(args: argparse.Namespace, base_dir: Path, candidates_dir: Path, compiled_files: list[Path]) -> Path:
    """Benchmark the top candidate files.

    Args:
        base_dir (Path): The base directory for the tuning process.
        candidates_dir (Path): The directory containing candidate files.
        compiled_files (list[Path]): A list of compiled files to benchmark.

    Returns:
        Path: The path to the log file containing the best benchmark results.
    """
    benchmark_results = []
    for compiled_file in compiled_files:
        result = run_command(args, f'./benchmark_dispatch.sh {compiled_file} {args.gpu_index_str}', check=False)
        benchmark_results.append(result.stdout)

    results_log = base_dir / "results.log"
    with results_log.open('w') as log_file:
        log_file.writelines(benchmark_results)
    
    best_results = []
    with results_log.open('r') as log_file:
        for line in log_file:
            if "failed" not in line:
                parts = line.split()
                best_results.append((parts[-1], f"{candidates_dir}/{parts[0]}.mlir", f"{candidates_dir}/configs/{parts[0]}_spec.mlir"))

    best_results = sorted(best_results, key=lambda x: x[0])[:20]
    best_log = base_dir / "best.log"
    with best_log.open('w') as log_file:
        for result in best_results:
            log_file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")
    
    return best_log

def compile_unet_candidates(args: argparse.Namespace, base_dir: Path, best_log: Path) -> list[str]:
    """Compile U-Net candidate files.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        base_dir (Path): The base directory for the tuning process.
        best_log (Path): The path to the log file containing the best benchmark results.

    Returns:
        list[str]: A list of U-Net candidate files.
    """
    p_pool = multiprocessing.Pool(processes=MAX_CONCURRENT_PROCESSES)
    results = []
    with best_log.open('r') as log_file:
        for line in log_file:
            if '/0.mlir' not in line:
                parts = line.split('\t')
                command = f"./compile_unet_candidate.sh {args.mode} {parts[2]}"
                # run_command(command)
                results.append(p_pool.apply_async(run_command, args=(args, command)))
                # Wait for at least one process to finish before adding more tasks
                if len(results) >= MAX_CONCURRENT_PROCESSES:
                    result = results[0].get()
                    results.pop(0)
    p_pool.close()

    # Wait for the remaining processes to complete
    for result in results:
        done = result.get()
    
    p_pool.join()

    unet_candidates = ["unet_baseline.vmfb"] + list(base_dir.glob("*.vmfb")) + ["unet_baseline.vmfb"]

    return unet_candidates

def benchmark_unet(args: argparse.Namespace, unet_candidates: list[str]) -> None:
    """Benchmark U-Net candidate files.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        unet_candidates (list[str]): A list of U-Net candidate files to benchmark.
    """
    for unet_candidate in unet_candidates:
        run_command(args, f"./benchmark_unet_candidate.sh {unet_candidate}")
        time.sleep(10)


def main():
    args = parse_arguments()

    base_dir = Path(f"tuning_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Setup logging\n")
    log_file_path = setup_logging(args, log_dir=base_dir)

    print("Gnerating candidates...")
    candidates, candidates_dir = generate_candidates(args, base_dir)
    print(f"Generated [{args.batch_size}] candidates in {candidates_dir}\n")

    print("Compiling candidates...")
    compiled_files, compiled_dir = compile_candidates(args, candidates, candidates_dir)
    print(f"Compiled files in {compiled_dir}\n")

    print("Benchmark and select top candidates...\n")
    best_log = benchmark_top_candidates(args, base_dir, candidates_dir, compiled_files)
    print("Compile unet candidtes...\n")
    unet_candidates = compile_unet_candidates(args, base_dir, best_log)
    print("Bnechmark unet candidtes..\n")
    benchmark_unet(args, unet_candidates)

    print("Done")

    print(log_file_path)

if __name__ == "__main__":
    main()
