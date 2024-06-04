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
import tune

"""
Sample Usage:

python autotune.py winograd 1286.mlir --lhs-dims=bmk --rhs-dims=bkn --tile-dims=*mnk --devices=1,3,5 --num-candidates=64

"""


# Default values for num_candidates and devices, change it as needed
DEFAULT_NUM_CANDIDATES = 1024
DEFAULT_DEVICE_LIST = [0]
# Default values for max number of workers
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count()
)  # the actual amount of worker that will be generated = max(min(max_cpu_workers, len(task_list)), 1)


def parse_devices(devices_str: str) -> list[int]:
    """Parse a comma-separated list of device IDs (e.g., "1,3,5" -> [1, 3, 5])."""
    try:
        devices = [int(device.strip()) for device in devices_str.split(",")]
        return devices
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid device list: {devices_str}. Error: {e}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune script")

    # Required arguments
    parser.add_argument(
        "mode", choices=["default", "winograd"], help="Compilation mode"
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the input benchmark file (.mlir)"
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=[0],
        help="Comma-separated list of device IDs (e.g., --devices=0,1). Default: [0]",
    )
    parser.add_argument(
        "--max-cpu-workers",
        type=int,
        default=DEFAULT_MAX_CPU_WORKERS,
        help=f"Max number of workers for CPU-bounding tasks (default: {DEFAULT_MAX_CPU_WORKERS}, the number of CPUs in current system)",
    )

    # tune.tune() options
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates to be generated by tune.py (default: {DEFAULT_NUM_CANDIDATES})",
    )
    parser.add_argument('--lhs-dims', help='Map of LHS matmul dims', type=str, default="mk")
    parser.add_argument('--rhs-dims', help='Map of RHS matmul dims', type=str, default="nk")
    parser.add_argument('--tile-dims', help='Map of tile size matmul dims', type=str, default="mnk")

    return parser.parse_args()


def setup_logging(args: argparse.Namespace, log_dir: Path) -> Path:
    log_file_name = f"autotune_{args.mode}_{args.input_file.stem}.log"
    log_file_path = log_dir / log_file_name

    handlers = [logging.FileHandler(log_file_path)]
    if args.verbose:
        handlers.append(logging.StreamHandler())
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # config logger in tune.py
    tune_logger = logging.getLogger("tune")
    tune_logger.setLevel(logging.DEBUG)

    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Devices: {args.devices}")
    logging.info(f"Number of candidates: {args.num_candidates}")
    logging.info(f"Extra options for tune.py: lhs-dims={args.lhs_dims}, rhs-dims{args.rhs_dims}, tile-dims={args.tile_dims}")
    logging.info(
        f"Device for Unet candidates: {args.devices[0]}"
    )  # Default use the first gpu from the user input --device list

    return log_file_path


def init_worker_context(queue: multiprocessing.Queue) -> None:
    """Assign a static index to current process as the worker ordinal, and specify the device indice to be used"""
    global worker_id, device_id

    worker_id, device_id = queue.get()


def create_worker_context_queue(device_ids: list[int]) -> multiprocessing.Queue:
    """Create queue contains Worker ID and Device ID for worker initialization"""
    worker_contexts_queue = multiprocessing.Manager().Queue()
    for worker_id, device_id in enumerate(device_ids):
        worker_contexts_queue.put((worker_id, device_id))

    return worker_contexts_queue


def worker_run_command_with_device_id(
    args: argparse.Namespace, command: list[str], check: bool = True
) -> subprocess.CompletedProcess:
    """worker add its device_id to the command for ./compile_unet_candidate.sh, return the run_command() result"""
    command.append(str(device_id))
    return run_command(args, command, check)


def run_command(
    args: argparse.Namespace, command: list[str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run a shell command and log the output.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        command (list): The command to run as a list of strings.
        check (bool, optional): Whether to check the command's exit status. Defaults to True.

    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    try:
        # Convert the command list to a command string for logging
        command_str = " ".join(command)
        if args.verbose:
            print(command_str)
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.stdout:
            logging.info(result.stdout)
        if result.stderr:
            logging.error(result.stderr)
            print(f"[warning] error flag raised by: {command_str}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(e.output)

        logging.error(
            f"Command '{command_str}' returned non-zero exit status {e.returncode}."
        )
        logging.error(f"Command '{command_str}' failed with error: {e.stderr}")
        if check:
            raise


def generate_candidates(
    args: argparse.Namespace, base_dir: Path
) -> tuple[list[Path], Path]:
    """Generate candidate files for tuning. Returns the list of candidate files and the candidates directory."""
    try:
        shutil.copy("config_prolog.mlir", base_dir / "config_prolog.mlir")
        shutil.copy("config_epilog.mlir", base_dir / "config_epilog.mlir")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        sys.exit(1)

    template_mlir = base_dir / "template.mlir"
    candidates_dir = base_dir / "candidates"

    shutil.copy(args.input_file, template_mlir)

    try:
        tune.tune(
            input=template_mlir,
            output=candidates_dir,
            limit=args.num_candidates,
            lhs_dims=args.lhs_dims,
            rhs_dims=args.rhs_dims,
            tile_dims=args.tile_dims
        )
        candidates = sorted(candidates_dir.glob("*.mlir"))
    except Exception as e:
        logging.error("An error occurred during candidates generation: %s", str(e))
        # Capture and log debug messages from tune.py
        tune_logger = logging.getLogger("tune")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                tune_logger.handlers.append(handler)
        tune_logger.exception("Error in tune.py:")

    candidates = sorted(candidates_dir.glob("*.mlir"))

    return candidates, candidates_dir


def compile_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    candidates: list[Path],
    candidate_dir: Path,
) -> tuple[list[Path], Path]:
    """Compile candidate files for tuning and record in candidate_vmfbs.txt. Returns the list of compiled files and the compiled files directory."""
    task_list = []
    for candidate in candidates:
        if "_config.mlir" not in candidate.name:
            command = ["./compile_candidate.sh", f"{args.mode}", f"{candidate}"]
            check = False
            task_list.append((args, command, check))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    with multiprocessing.Pool(num_worker) as worker_pool:
        worker_pool.starmap(run_command, task_list)

    compiled_dir = candidate_dir / "compiled"
    compiled_files = sorted(compiled_dir.glob("*.vmfb"))

    # Write compiled files to candidate_vmfbs.txt
    candidate_vmfbs_file = base_dir / "candidate_vmfbs.txt"
    with candidate_vmfbs_file.open("w") as f:
        for compiled_file in compiled_files:
            f.write(f"{compiled_file}\n")

    return compiled_files, compiled_dir


def benchmark_top_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    candidates_dir: Path,
    compiled_files: list[Path],
) -> Path:
    """Benchmark the candidate files and store the top20 results in file (best.log). Return the log file"""
    task_list = []
    for compiled_file in compiled_files:
        command = ["./benchmark_dispatch.sh", f"{compiled_file}"]
        check = False
        task_list.append((args, command, check))

    worker_context_queue = create_worker_context_queue(args.devices)
    with multiprocessing.Pool(
        len(args.devices), init_worker_context, (worker_context_queue,)
    ) as worker_pool:
        results = worker_pool.starmap(worker_run_command_with_device_id, task_list)

    benchmark_results = [result.stdout for result in results]

    results_log = base_dir / "results.log"
    with results_log.open("w") as log_file:
        log_file.writelines(benchmark_results)

    best_results = []
    with results_log.open("r") as log_file:
        for line in log_file:
            if "failed" not in line:
                parts = line.split()
                best_results.append(
                    (
                        parts[-1],
                        f"{candidates_dir}/{parts[0]}.mlir",
                        f"{candidates_dir}/configs/{parts[0]}_spec.mlir",
                    )
                )

    best_results = sorted(best_results, key=lambda x: x[0])[:20]
    best_log = base_dir / "best.log"
    with best_log.open("w") as log_file:
        for result in best_results:
            log_file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")

    return best_log


def compile_unet_candidates(
    args: argparse.Namespace, base_dir: Path, best_log: Path
) -> list[str]:
    """Compile U-Net candidates stored in best.log. Return the list of U-Net candidate files."""
    task_list = []
    with best_log.open("r") as log_file:
        for line in log_file:
            if "/0.mlir" not in line:
                command = [
                    "./compile_unet_candidate.sh",
                    f"{args.mode}",
                    f"{args.devices[0]}",
                ]  # Default use the first gpu from the user input --device list
                task_list.append((args, command))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    with multiprocessing.Pool(num_worker) as worker_pool:
        worker_pool.starmap(run_command, task_list)

    unet_candidates = (
        ["unet_baseline.vmfb"] + list(base_dir.glob("*.vmfb")) + ["unet_baseline.vmfb"]
    )

    return unet_candidates


def benchmark_unet(
    args: argparse.Namespace, base_dir: Path, unet_candidates: list[str]
) -> None:
    """Benchmark U-Net candidate files and log the results. Return the file path of unet_results.log"""
    unet_result_log = base_dir / "unet_results.log"

    with unet_result_log.open("w") as log_file:
        for unet_candidate in unet_candidates:
            command = [
                "./benchmark_unet_candidate.sh",
                f"{unet_candidate}",
                f"{args.devices[0]}",
            ]  # Default use the first gpu from the user input --device list
            result = run_command(args, command)
            log_file.write(result.stdout)
            log_file.write(result.stderr)
            if result.returncode != 0:
                print("Failed")
            time.sleep(10)

    return unet_result_log


def main():
    args = parse_arguments()

    base_dir = Path(f"tuning_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Setup logging\n")
    log_file_path = setup_logging(args, log_dir=base_dir)

    print("Gnerating candidates...")
    candidates, candidates_dir = generate_candidates(args, base_dir)
    print(f"Generated [{args.num_candidates}] candidates in {candidates_dir}\n")

    print("Compiling candidates...")
    compiled_files, compiled_dir = compile_candidates(
        args, base_dir, candidates, candidates_dir
    )
    print(f"Compiled files in {compiled_dir}\n")

    print("Benchmarking top candidates...")
    best_log = benchmark_top_candidates(args, base_dir, candidates_dir, compiled_files)
    print(f"Top20 candidates selected and stored in {best_log}\n")

    print("Compiling unet candidtes...")
    unet_candidates = compile_unet_candidates(args, base_dir, best_log)
    print("Unet candidtes compiled\n")

    print("Bnechmarking unet candidtes...")
    unet_result_log = benchmark_unet(args, base_dir, unet_candidates)
    print(f"Done, stored unet result in {unet_result_log}\n")

    print("Check result in log file:")
    print(log_file_path, end="\n")


if __name__ == "__main__":
    main()
