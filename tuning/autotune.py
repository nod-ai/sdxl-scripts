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
from tqdm import tqdm
import re
import hashlib
from dataclasses import dataclass
from typing import Literal
import pickle

"""
Sample Usage:

python autotune.py winograd 1286.mlir --lhs-dims=bmk --rhs-dims=bkn --tile-dims=*mnk --devices=1,3,5 --num-candidates=64

"""


# Default values for num_candidates and devices, change it as needed
DEFAULT_NUM_CANDIDATES = 2048
DEFAULT_DEVICE_LIST = [0]

# Default values for max number of workers
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count() // 2
)  # the actual amount of worker that will be generated = max(min(max_cpu_workers//2, len(task_list)), 1)
"""note: Do not use all CPU cores"""


@dataclass
class CandidateTracker:
    candidate_id: int
    mlir_path: str = None
    mlir_config_path: str = None
    configuration: tune.Configuration = None
    compilation_successful: bool = None
    compiled_vmfb_path: str = None
    first_benchmark_time: float = None
    unet_candidate_path: str = None
    unet_vmfb_hash: str = None
    unet_benchmark_time: float = None


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
        default=DEFAULT_DEVICE_LIST,
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
    parser.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    parser.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    parser.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )

    return parser.parse_args()


def setup_logging(args: argparse.Namespace, log_dir: Path) -> Path:
    log_file_name = f"autotune_{args.mode}_{args.input_file.stem}.log"
    log_file_path = log_dir / log_file_name

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create stream handler for logging to the console (only warnings and higher)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Create a formatter that dynamically adds [levelname] for ERROR and WARNING
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.CRITICAL:
                return f"{record.message}"
            else:
                return f"[{record.levelname}] {record.message}"

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = CustomFormatter()

    # Set formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger to the lowest level
        handlers=[file_handler, console_handler],
    )

    # If verbose flag is set, add a console handler for INFO level and higher
    if args.verbose:
        verbose_console_handler = logging.StreamHandler()
        verbose_console_handler.setLevel(logging.INFO)
        verbose_console_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(verbose_console_handler)

    # config logger in tune.py
    tune_logger = logging.getLogger("tune")
    tune_logger.setLevel(logging.DEBUG)

    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Devices: {args.devices}")
    logging.info(f"Number of candidates: {args.num_candidates}")
    logging.info(
        f"Extra options for tune.py: lhs-dims={args.lhs_dims}, rhs-dims={args.rhs_dims}, tile-dims={args.tile_dims}"
    )
    logging.info(
        f"Device for Unet candidates: {args.devices[0]}"
    )  # Default use the first gpu from the user input --device list

    return log_file_path


def handle_error(
    condition: bool,
    msg: str,
    level: int = logging.ERROR,
    exit_program: bool = True,
) -> None:
    """Handles errors with logging and optional program exit"""
    if not condition:
        return

    # Log the message with the specified level
    if level == logging.ERROR:
        logging.error(msg)
    elif level == logging.WARNING:
        logging.warning(msg)
    elif level == logging.INFO:
        logging.info(msg)
    elif level == logging.DEBUG:
        logging.debug(msg)
    else:
        raise ValueError(
            "Invalid logging level specified: choose from logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG"
        )

    if exit_program:
        sys.exit(1)


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
    task_tuple: tuple[argparse.Namespace, str, bool]
) -> subprocess.CompletedProcess:
    """worker add its device_id to the command for ./compile_unet_candidate.sh, return the run_command() result"""
    args, command, check = task_tuple
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
        logging.debug(f"Run: {command_str}")
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.stdout:
            logging.info(f"stdout: {result.stdout}")
        if result.stderr:
            logging.error(f"stderr: {result.stderr}")
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


def run_command_wrapper(
    task_tuple: tuple[argparse.Namespace, str, bool]
) -> subprocess.CompletedProcess:
    """pool.imap_unordered can't iterate an iterable of iterables input, this function helps dividing arguments"""
    args, command, check = task_tuple
    return run_command(args, command, check)


def multiprocess_progress_wrapper(
    num_worker: int,
    task_list: list,
    function: callable,
    initializer: callable = None,
    initializer_inputs=None,
) -> list[subprocess.CompletedProcess]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    # Create a multiprocessing pool
    with multiprocessing.Pool(
        num_worker, initializer, initializer_inputs
    ) as worker_pool:
        # Use tqdm to create a progress bar
        with tqdm(total=len(task_list)) as pbar:
            # Use imap_unordered to asynchronously execute the worker function on each task
            for result in worker_pool.imap_unordered(function, task_list):
                pbar.update(1)  # Update progress bar
                results.append(result)
    return results


def numerical_sort_key(path: Path) -> tuple[int, str]:
    """
    Define a sort key function that splits the filename into a numeric and a string part.
    Order: 0 | 0_a | 0_b | 1 | 1_a | 2
    """
    # Extract the numeric part at the start of the filename
    match = re.match(r"(\d+)", path.stem)
    if match:
        numeric_part = int(match.group(1))
        # The rest of the filename after the numeric part
        remaining_part = path.stem[len(match.group(0)) :]
    else:
        numeric_part = float("inf")
        remaining_part = path.stem
    return (numeric_part, remaining_part)


def calculate_md5(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def find_collisions(
    hash_list: list[tuple[int, str]]
) -> tuple[bool, list[tuple[str, list[int]]]]:
    """
    Detect hash value collisions
    Take input list of candidate index numbers and hash value strings: ex. [(1, 'abc'), (2, 'def'), (3, 'abc')]
    Return collision boolean value and list of unique hash values along with their corresponding indices: ex. [('abc', [1,3]), ('def', [2])]
    """
    hash_count = {}

    # Count occurrences of each hash_val
    for index, hash_val in hash_list:
        if hash_val in hash_count:
            hash_count[hash_val].append(index)
        else:
            hash_count[hash_val] = [index]

    # Prepare output for all hash values
    hash_values = [(hash_val, indices) for hash_val, indices in hash_count.items()]

    # Determine if there are collisions
    collisions_exist = any(len(indices) > 1 for hash_val, indices in hash_count.items())

    return collisions_exist, hash_values


def load_pickle(file_path: Path) -> list[any]:
    with open(file_path, "rb") as file:
        loaded_array = pickle.load(file)
    return loaded_array


def generate_candidates(
    args: argparse.Namespace, base_dir: Path, candidate_trackers: CandidateTracker
) -> tuple[list[Path], Path]:
    """Generate candidate files for tuning. Returns the list of candidate files and the candidates directory."""
    logging.info("generate_candidates()")

    try:
        shutil.copy("config_prolog.mlir", base_dir / "config_prolog.mlir")
        shutil.copy("config_epilog.mlir", base_dir / "config_epilog.mlir")
    except FileNotFoundError as e:
        handle_error(condition=True, msg=f"Configuration file not found: {e}")

    template_mlir = base_dir / "template.mlir"
    candidates_dir = base_dir / "candidates"

    shutil.copy(args.input_file, template_mlir)

    mlirs = []
    try:
        logging.debug("Captured messages from tune.py:")
        tune.tune(
            input=template_mlir,
            output=candidates_dir,
            limit=args.num_candidates,
            lhs_dims=args.lhs_dims,
            rhs_dims=args.rhs_dims,
            tile_dims=args.tile_dims,
        )
        mlirs = sorted(candidates_dir.glob("*.mlir"), key=numerical_sort_key)
    except Exception as e:
        logging.error("An error occurred during candidates generation: %s", str(e))
        # Capture and log debug messages from tune.py
        tune_logger = logging.getLogger("tune")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                tune_logger.handlers.append(handler)
        tune_logger.exception("Error in tune.py:")
    logging.debug("tune.py ends")

    candidate_configs = load_pickle(candidates_dir / "configs.pkl")
    candidate_configs.insert(0, None)  # No Configuration class for 0.mlir

    # Create candidate trackers
    assert len(mlirs) // 2 + 1 == len(candidate_configs)
    candidates = []
    for mlir in mlirs:
        if "_config.mlir" not in mlir.name:
            candidates.append(mlir)
            new_candidate = CandidateTracker(
                candidate_id=mlir.stem,
                mlir_path=mlir,
                configuration=candidate_configs[int(mlir.stem)],
            )
            candidate_trackers.append(new_candidate)
        else:
            candidate_trackers[
                int(mlir.stem.split("_config")[0])
            ].mlir_config_path = mlir

    handle_error(
        condition=(len(candidates) == 0), msg="Failed to generate any candidates"
    )

    return candidates, candidates_dir


def compile_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    candidates: list[Path],
    candidate_dir: Path,
    candidate_trackers: CandidateTracker,
) -> tuple[list[Path], Path]:
    """Compile candidate files for tuning and record in candidate_vmfbs.txt. Returns the list of compiled files and the compiled files directory."""
    logging.info("compile_candidates()")

    task_list = []
    for candidate in candidates:
        command = ["./compile_candidate.sh", f"{args.mode}", f"{candidate}"]
        check = False
        task_list.append((args, command, check))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    compiled_dir = candidate_dir / "compiled"
    compiled_files = sorted(compiled_dir.glob("*.vmfb"), key=numerical_sort_key)
    failed_dir = candidate_dir / "failed"
    failed_files = sorted(failed_dir.glob("*.mlir"), key=numerical_sort_key)

    total, good, bad = len(task_list), len(compiled_files), len(failed_files)
    compiling_rate = good / total * 100
    logging.critical(
        f"Total: {total} | Compiled: {good} | Failed: {bad} | Compiling Rate: {compiling_rate:.1f}%"
    )

    # Write compiled files to candidate_vmfbs.txt
    candidate_vmfbs_file = base_dir / "candidate_vmfbs.txt"
    with candidate_vmfbs_file.open("w") as f:
        for compiled_file in compiled_files:
            f.write(f"{compiled_file}\n")

    # Update candidate tracker
    for failed_file in failed_files:
        index = int(failed_file.stem)
        candidate_trackers[index].compilation_successful = False
    for compiled_file in compiled_files:
        index = int(compiled_file.stem)
        candidate_trackers[index].compilation_successful = True
        candidate_trackers[index].compiled_vmfb_path = compiled_file

    handle_error(
        condition=(good == 0),
        msg="Failed to compile all candidate .mlir files",
        exit_program=False,
    )
    handle_error(
        condition=(compiling_rate < 10),
        msg=f"Compiling rate [{compiling_rate:.1f}%] < 10%",
        level=logging.WARNING,
        exit_program=False,
    )

    return compiled_files, compiled_dir


def benchmark_top_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    candidates_dir: Path,
    compiled_files: list[Path],
    candidate_trackers: CandidateTracker,
) -> Path:
    """Benchmark the candidate files and store the top20 results in file (best.log). Return the log file"""
    logging.info("benchmark_top_candidates()")

    task_list = []
    for compiled_file in compiled_files:
        command = ["./benchmark_dispatch.sh", f"{compiled_file}"]
        check = False
        task_list.append((args, command, check))

    worker_context_queue = create_worker_context_queue(args.devices)
    results = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=task_list,
        function=worker_run_command_with_device_id,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )

    benchmark_results = [result.stdout for result in results]

    results_log = base_dir / "results.log"
    with results_log.open("w") as log_file:
        log_file.writelines(benchmark_results)

    best_results = []
    with results_log.open("r") as log_file:
        for line in log_file:
            if "failed" not in line:
                parts = line.split()

                # Update candidate tracker
                candidate_trackers[int(parts[0])].first_benchmark_time = float(
                    parts[-1]
                )

                best_results.append(
                    (
                        parts[-1],
                        f"{candidates_dir}/{parts[0]}.mlir",
                        f"{candidates_dir}/configs/{parts[0]}_spec.mlir",
                    )
                )

    benchmarked_dir = candidates_dir / "compiled"
    benchmarked_files = sorted(benchmarked_dir.glob("*.vmfb"), key=numerical_sort_key)
    benchmark_failed_dir = benchmarked_dir / "benchmark_failed"
    benchmark_failed_files = sorted(
        benchmark_failed_dir.glob("*.vmfb"), key=numerical_sort_key
    )

    logging.critical(
        f"Total: {len(benchmark_results)} | Benchmarked: {len(benchmarked_files)} | Failed: {len(benchmark_failed_files)} | Benchmarking Rate: {(len(benchmarked_files)/len(benchmark_results))*100}%"
    )

    handle_error(
        condition=(len(best_results) == 0),
        msg="Failed to benchmark all candidate .vmfb files",
    )

    best_results = sorted(best_results, key=lambda x: x[0])[:20]
    best_log = base_dir / "best.log"
    with best_log.open("w") as log_file:
        for result in best_results:
            log_file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")

    return best_log


def compile_unet_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    best_log: Path,
    candidate_trackers: CandidateTracker,
) -> list[str]:
    """Compile U-Net candidates stored in best.log. Return the list of U-Net candidate files."""
    logging.info("compile_unet_candidates()")

    task_list = []
    with best_log.open("r") as log_file:
        for line in log_file:
            if "/0.mlir" not in line:
                input_file = line.strip().split()[2]
                command = [
                    "./compile_unet_candidate.sh",
                    f"{args.mode}",
                    f"{input_file}",
                ]
                check = True
                task_list.append((args, command, check))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    unet_candidates = list(base_dir.glob("*.vmfb"))

    unet_candidates_hash_list = []

    # Update candidate tracker
    for unet_candidate in unet_candidates:
        index = int(unet_candidate.stem.split("_")[-1])
        candidate_trackers[index].unet_candidate_path = unet_candidate
        hash_val = calculate_md5(candidate_trackers[index].unet_candidate_path)
        candidate_trackers[index].unet_vmfb_hash = hash_val
        unet_candidates_hash_list.append((index, hash_val))

    # Check if unet candidate produces tbe same .vmfb
    collision_detected, hash_list = find_collisions(unet_candidates_hash_list)
    if collision_detected:
        unique_unet_candidates = []
        logging.warning("Collisions detected")
        for hash_val, indices in hash_list:
            if len(indices) != 1:
                logging.warning(
                    f"Hash value '{hash_val}' collided at candidate {indices}."
                )
            unique_unet_candidates.append(
                candidate_trackers[indices[0]].unet_candidate_path
            )

    return unique_unet_candidates if collision_detected else unet_candidates


def benchmark_unet(
    args: argparse.Namespace,
    base_dir: Path,
    unet_candidates: list[str],
    candidate_trackers: CandidateTracker,
) -> None:
    """Benchmark U-Net candidate files and log the results. Return the file path of unet_results.log"""
    logging.info("benchmark_unet()")

    unet_candidates = ["unet_baseline.vmfb"] + unet_candidates + ["unet_baseline.vmfb"]
    # Update candidate tracker
    candidate_trackers[0].unet_candidate_path = "unet_baseline.vmfb"

    unet_result_log = base_dir / "unet_results.log"

    with unet_result_log.open("w") as log_file:
        with tqdm(total=len(unet_candidates)) as pbar:
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
                    logging.error(f"Failed: {command}")
                else:
                    # Update candidate tracker
                    # ex. ['Benchmarking:', '/sdxl-scripts/tuning/unet_baseline.vmfb', 'on', 'device', '4', 'BM_main/process_time/real_time_median', '65.3', 'ms', '66.7', 'ms', '5', 'items_per_second=15.3201/s']
                    parts = result.stdout.split()
                    if "unet_baseline.vmfb" in parts[1]:
                        candidate_trackers[0].unet_benchmark_time = (
                            float(parts[6])
                            if candidate_trackers[0].unet_benchmark_time is None
                            or float(parts[6])
                            < candidate_trackers[0].unet_benchmark_time
                            else candidate_trackers[0].unet_benchmark_time
                        )
                    else:
                        candidate_trackers[
                            int(parts[1].split("_")[-1].split(".")[0])
                        ].unet_benchmark_time = float(parts[6])
                time.sleep(10)
                pbar.update(1)

    return unet_result_log


def main():
    args = parse_arguments()

    base_dir = Path(f"tuning_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate_trackers: list[CandidateTracker] = []

    print("Setup logging")
    log_file_path = setup_logging(args, log_dir=base_dir)
    print(log_file_path, end="\n\n")

    print("Generating candidates...")
    candidates, candidates_dir = generate_candidates(args, base_dir, candidate_trackers)
    print(f"Generated [{len(candidates)}] candidates in {candidates_dir}\n")

    print("Compiling candidates...")
    compiled_files, compiled_dir = compile_candidates(
        args, base_dir, candidates, candidates_dir, candidate_trackers
    )
    print(f"Compiled [{len(compiled_files)}] files in {compiled_dir}\n")

    print("Benchmarking top candidates...")
    best_log = benchmark_top_candidates(
        args, base_dir, candidates_dir, compiled_files, candidate_trackers
    )
    print(f"Top candidates results are stored in {best_log}\n")

    print("Compiling unet candidates...")
    unet_candidates = compile_unet_candidates(
        args, base_dir, best_log, candidate_trackers
    )
    print(f"Unet candidates compiled in {base_dir}\n")

    print("Bnechmarking unet candidates...")
    unet_result_log = benchmark_unet(
        args, base_dir, unet_candidates, candidate_trackers
    )
    print(f"Done, stored unet result in {unet_result_log}\n")

    candidate_trackers_file_path = base_dir / "candidate_trackers.pkl"
    with open(candidate_trackers_file_path, "wb") as file:
        pickle.dump(candidate_trackers, file)
    print(f"Candidate trackers are saved in {candidate_trackers_file_path}")

    print("Check the detailed log in:")
    print(log_file_path)

    for candidate in candidate_trackers:
        logging.debug(candidate)
        if args.verbose:
            print(candidate)


if __name__ == "__main__":
    main()
