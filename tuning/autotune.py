#!/usr/bin/env python3

import sys
import shutil
import subprocess
import logging
import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path
import time
import multiprocessing
import queue
import tune
from tqdm import tqdm
import re
import hashlib
from dataclasses import dataclass
from typing import Type, Optional, Callable, Iterable, Any
import pickle
from itertools import groupby

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
    mlir_path: Optional[Path] = None
    mlir_config_path: Optional[Path] = None
    configuration: Optional[tune.Configuration] = None
    compilation_successful: Optional[bool] = None
    compiled_vmfb_path: Optional[Path] = None
    first_benchmark_time: Optional[float] = None
    first_benchmark_device_id: Optional[int] = None
    unet_candidate_path: Optional[Path] = None
    unet_vmfb_hash: Optional[str] = None
    unet_benchmark_time: Optional[float] = None
    unet_benchmark_device_id: Optional[int] = None
    baseline_benchmark_time: Optional[float] = None
    calibrated_benchmark_diff: Optional[float] = None


@dataclass
class TaskTuple:
    args: argparse.Namespace
    command: list[str]
    check: bool = True
    command_need_device_id: bool = False
    cooling_time: int = 0
    result_need_device_id: bool = False


@dataclass
class TaskResult:
    result: subprocess.CompletedProcess
    device_id: int = None


@dataclass
class BenchmarkOutput:
    output_str: Optional[str] = None

    @property
    def output_list(self) -> list[str]:
        # e.g. ['Benchmarking:', '/sdxl-scripts/tuning/tuning_2024_07_19_08_55/unet_candidate_12.vmfb', 'on', 'device', '4', 'BM_main/process_time/real_time_median', '65.3', 'ms', '66.7', 'ms', '5', 'items_per_second=15.3201/s']
        if self.output_str is None:
            return []
        return self.output_str.split()

    @property
    def unet_candidate_path(self) -> Optional[str]:
        if not self.output_list or len(self.output_list) < 2:
            return None
        return self.output_list[1]

    @property
    def candidate_id(self) -> Optional[int]:
        if self.unet_candidate_path:
            try:
                return int(self.unet_candidate_path.split("_")[-1].split(".")[0])
            except ValueError:
                return None
        return None

    @property
    def device_id(self) -> Optional[str]:
        if len(self.output_list) < 5:
            return None
        return self.output_list[4]

    @property
    def benchmark_time(self) -> Optional[float]:
        if len(self.output_list) < 7:
            return None
        try:
            return float(self.output_list[6])
        except ValueError:
            return None

    def calibrated_output_str(self, change: float) -> str:
        if self.output_str is None:
            return self.output_str

        benchmark_time = self.benchmark_time
        if benchmark_time is None:
            return self.output_str

        # Calculate the percentage change
        percentage_change = change * 100
        new_benchmark_time = benchmark_time + (benchmark_time * change)

        # Format the change to be added to the string
        change_str = f"({percentage_change:+.3f}%)"

        # Use regex to find and replace the old benchmark time with the new one
        new_output_str = re.sub(
            r"(\d+(\.\d+)?)\s*ms",
            lambda m: f"{new_benchmark_time} ms {change_str}",
            self.output_str,
            count=1,
        )

        return new_output_str


def parse_devices(devices_str: str) -> list[int]:
    """Parse a comma-separated list of device IDs (e.g., "1,3,5" -> [1, 3, 5])."""
    devices = []
    try:
        devices = [int(device.strip()) for device in devices_str.split(",")]
    except ValueError as e:
        handle_error(
            condition=True,
            msg=f"Invalid device list: {devices_str}. Error: {e}",
            error_type=argparse.ArgumentTypeError,
        )
    return devices


class ExecutionPhases(str, Enum):
    dont_stop = ""
    generate_candidates = "generate-candidates"
    compile_candidates = "compile-candidates"
    benchmark_candidates = "benchmark-candidates"
    compile_unet_candidates = "compile-unet-candidates"
    benchmark_unet_candidates = "benchmark-unet-candidates"


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
    parser.add_argument(
        "--stop-after",
        choices=[x.value for x in ExecutionPhases],
        default=ExecutionPhases.dont_stop,
        help="Stop execution after specified phase",
    )
    parser.add_argument(
        "--num-unet-candidates",
        help="Maximum number of stage 2 candidates",
        type=int,
        default=50,
    )

    # tune.tune() options
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates to be generated by tune.py (default: {DEFAULT_NUM_CANDIDATES})",
    )
    parser.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use",
        type=int,
        default=4,
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
    error_type: Type[BaseException] = Exception,
    exit_program: bool = False,
) -> None:
    """Handles errors with logging and optional program exit"""
    if not condition:
        return

    # Log the message with the specified level
    if level == logging.ERROR:
        logging.error(msg)
        raise error_type(msg)
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


def create_worker_context_queue(device_ids: list[int]) -> queue.Queue[tuple[int, int]]:
    """Create queue contains Worker ID and Device ID for worker initialization"""
    worker_contexts_queue = multiprocessing.Manager().Queue()
    for worker_id, device_id in enumerate(device_ids):
        worker_contexts_queue.put((worker_id, device_id))

    return worker_contexts_queue


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
    result = None
    try:
        # Convert the command list to a command string for logging
        command_str = " ".join(command)
        logging.debug(f"Run: {command_str}")
        result = subprocess.run(command, check=check, capture_output=True, text=True)
        if result.stdout:
            logging.info(f"stdout: {result.stdout}")
        if result.stderr:
            logging.error(f"stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        print(e.output)

        logging.error(
            f"Command '{command_str}' returned non-zero exit status {e.returncode}."
        )
        logging.error(f"Command '{command_str}' failed with error: {e.stderr}")
        if check:
            raise
    except KeyboardInterrupt:
        print("Ctrl+C detected, terminating child processes...")

    return result


def run_command_wrapper(task_tuple: TaskTuple) -> TaskResult:
    """pool.imap_unordered can't iterate an iterable of iterables input, this function helps dividing arguments"""
    if task_tuple.command_need_device_id:
        # worker add its device_id to the end of command list
        task_tuple.command.append(str(device_id))

    task_result = TaskResult(
        run_command(task_tuple.args, task_tuple.command, task_tuple.check)
    )
    task_result.device_id = device_id if task_tuple.result_need_device_id else None

    time.sleep(task_tuple.cooling_time)

    return task_result


def multiprocess_progress_wrapper(
    num_worker: int,
    task_list: list,
    function: Callable,
    initializer: Optional[Callable] = None,
    initializer_inputs: Optional[Iterable[Any]] = None,
) -> list[subprocess.CompletedProcess]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    # Create a multiprocessing pool
    with multiprocessing.Pool(
        num_worker, initializer, initializer_inputs
    ) as worker_pool:
        # Use tqdm to create a progress bar
        with tqdm(total=len(task_list)) as pbar:
            try:
                # Use imap_unordered to asynchronously execute the worker function on each task
                for result in worker_pool.imap_unordered(function, task_list):
                    pbar.update(1)  # Update progress bar
                    results.append(result)
            except KeyboardInterrupt:
                # If Ctrl+C is pressed, terminate all child processes
                worker_pool.terminate()
                worker_pool.join()
                sys.exit(1)  # Exit the script
            except:
                assert False
    return results


def numerical_sort_key(path: Path) -> tuple[int | float, str]:
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


def calculate_md5(file_path: Path) -> str:
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


def load_pickle(file_path: Path) -> list[Any]:
    handle_error(
        condition=(not file_path.exists()),
        msg=f"Configuration file not found: {file_path}",
        error_type=FileNotFoundError,
    )
    with open(file_path, "rb") as file:
        loaded_array = pickle.load(file)
    return loaded_array


def generate_candidates(
    args: argparse.Namespace, base_dir: Path, candidate_trackers: list[CandidateTracker]
) -> tuple[list[Path], Path]:
    """Generate candidate files for tuning. Returns the list of candidate files and the candidates directory."""
    logging.info("generate_candidates()")

    try:
        shutil.copy("config_prolog.mlir", base_dir / "config_prolog.mlir")
        shutil.copy("config_epilog.mlir", base_dir / "config_epilog.mlir")
    except FileNotFoundError as e:
        handle_error(
            condition=True,
            msg=f"Configuration file not found: {e}",
            error_type=FileNotFoundError,
        )

    template_mlir = base_dir / "template.mlir"
    candidates_dir = base_dir / "candidates"

    shutil.copy(args.input_file, template_mlir)

    mlirs = []
    try:
        logging.debug("Captured messages from tune.py:")
        tune.tune(
            input=str(template_mlir),
            output=str(candidates_dir),
            limit=args.num_candidates,
            num_subgroups=args.num_subgroups,
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
        raise
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
                candidate_id=int(mlir.stem),
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
    candidate_trackers: list[CandidateTracker],
) -> tuple[list[Path], Path]:
    """Compile candidate files for tuning and record in candidate_vmfbs.txt. Returns the list of compiled files and the compiled files directory."""
    logging.info("compile_candidates()")

    task_list = []
    for candidate in candidates:
        command = ["./compile_candidate.sh", f"{args.mode}", f"{candidate}"]
        task_list.append(TaskTuple(args, command, check=False))

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
        condition=(good == 0), msg="Failed to compile all candidate .mlir files"
    )
    handle_error(
        condition=(compiling_rate < 10),
        msg=f"Compiling rate [{compiling_rate:.1f}%] < 10%",
        level=logging.WARNING,
    )

    return compiled_files, compiled_dir


def benchmark_compiled_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    candidates_dir: Path,
    compiled_files: list[Path],
    candidate_trackers: list[CandidateTracker],
) -> Path:
    """Benchmark the candidate files and store the topN results in file (best.log). Return the log file"""
    logging.info("benchmark_top_candidates()")

    task_list = []
    for compiled_file in compiled_files:
        command = ["./benchmark_dispatch.sh", f"{compiled_file}"]
        task_list.append(
            TaskTuple(args, command, check=False, command_need_device_id=True)
        )

    worker_context_queue = create_worker_context_queue(args.devices)
    task_results = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=task_list,
        function=run_command_wrapper,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )

    benchmark_results = [task_result.result.stdout for task_result in task_results]

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

    benchmarking_rate = (len(benchmarked_files) / len(benchmark_results)) * 100
    logging.critical(
        f"Total: {len(benchmark_results)} | Benchmarked: {len(benchmarked_files)} | Failed: {len(benchmark_failed_files)} | Benchmarking Rate: {benchmarking_rate:.1f}%"
    )

    handle_error(
        condition=(len(best_results) == 0),
        msg="Failed to benchmark all candidate .vmfb files",
    )

    best_results = sorted(best_results, key=lambda x: float(x[0]))[
        : args.num_unet_candidates
    ]
    best_log = base_dir / "best.log"
    with best_log.open("w") as log_file:
        for result in best_results:
            log_file.write(f"{result[0]}\t{result[1]}\t{result[2]}\n")

    return best_log


def compile_unet_candidates(
    args: argparse.Namespace,
    base_dir: Path,
    best_log: Path,
    candidate_trackers: list[CandidateTracker],
) -> list[int]:
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
                task_list.append(TaskTuple(args, command))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    unet_candidates_files = list(base_dir.glob("*.vmfb"))

    unet_candidates_indexes = []
    unet_candidates_hash_list = []

    # Update candidate tracker
    for unet_candidate in unet_candidates_files:
        index = int(unet_candidate.stem.split("_")[-1])
        candidate_trackers[index].unet_candidate_path = unet_candidate
        hash_val = calculate_md5(candidate_trackers[index].unet_candidate_path)
        candidate_trackers[index].unet_vmfb_hash = hash_val
        unet_candidates_hash_list.append((index, hash_val))
        unet_candidates_indexes.append(index)

    # Check if unet candidate produces tbe same .vmfb
    collision_detected, hash_list = find_collisions(unet_candidates_hash_list)
    if collision_detected:
        unique_unet_candidates_indexes = []
        logging.warning("Collisions detected")
        for hash_val, indices in hash_list:
            if len(indices) != 1:
                logging.warning(
                    f"Hash value '{hash_val}' collided at candidate {indices}."
                )
            unique_unet_candidates_indexes.append(indices[0])

    return (
        unique_unet_candidates_indexes
        if collision_detected
        else unet_candidates_indexes
    )


def sort_candidates_by_first_benchmark_times(
    candidate_indexes: list[int], candidate_trackers: CandidateTracker
) -> list[int]:
    """Sorts candidate indexes based on their first benchmark times in ascending order"""
    first_benchmark_times = [
        candidate_trackers[index].first_benchmark_time for index in candidate_indexes
    ]
    combined = list(zip(candidate_indexes, first_benchmark_times))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    sorted_indexes, _ = zip(*combined_sorted)
    sorted_indexes = list(sorted_indexes)
    return sorted_indexes


def group_benchmark_results_by_device_id(
    benchmark_results: list[TaskResult],
) -> list[list[TaskResult]]:
    """
    Groups benchmark results by device ID.

    e.g.
    [TaskResult(res1, device_1), TaskResult(res2, device_2), TaskResult(res3, device_1)]
    ----->
    [ [TaskResult(res1, device_1), TaskResult(res3, device_1)], [TaskResult(res2, device_2)] ]
    """
    grouped_results = [
        list(group)
        for _, group in groupby(benchmark_results, key=lambda tr: tr.device_id)
    ]
    grouped_results: dict[int, list[TaskResult]] = {}
    for result in benchmark_results:
        if result.device_id not in grouped_results:
            grouped_results[result.device_id] = []
        grouped_results[result.device_id].append(result)

    grouped_benchmark_results = [
        grouped_results[device_id] for device_id in sorted(grouped_results)
    ]

    return grouped_benchmark_results


def parse_grouped_benchmark_results(
    grouped_benchmark_results: list[list[TaskResult]],
    candidate_trackers: CandidateTracker,
) -> list[str]:
    """Update candidate_trackers and collect strings"""
    dump_list = []

    for same_device_results in grouped_benchmark_results:
        for unet_candidate_result in same_device_results:
            res = BenchmarkOutput(unet_candidate_result.result.stdout)
            if "unet_baseline.vmfb" in res.unet_candidate_path:
                baseline_time = res.benchmark_time
                dump_list.append(res.output_str)
                continue
            candidate_trackers[
                res.candidate_id
            ].unet_benchmark_time = res.benchmark_time
            candidate_trackers[res.candidate_id].baseline_benchmark_time = baseline_time
            candidate_trackers[
                res.candidate_id
            ].unet_benchmark_device_id = res.device_id
            candidate_trackers[res.candidate_id].calibrated_benchmark_diff = (
                res.benchmark_time - baseline_time
            ) / baseline_time
            dump_str = res.calibrated_output_str(
                candidate_trackers[res.candidate_id].calibrated_benchmark_diff
            )

            dump_list.append(dump_str)

    return dump_list


def benchmark_unet(
    args: argparse.Namespace,
    base_dir: Path,
    unet_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
) -> Path:
    """Benchmark U-Net candidate files and log the results. Return the file path of unet_results.log"""
    logging.info("benchmark_unet()")

    unet_result_log = base_dir / "unet_results.log"
    unet_baseline_filepath = Path("./unet_baseline.vmfb")
    candidate_trackers[0].unet_candidate_path = unet_baseline_filepath

    unet_candidates = sort_candidates_by_first_benchmark_times(
        unet_candidates, candidate_trackers
    )

    # Benchmarking unet candidates
    worker_context_queue = create_worker_context_queue(args.devices)
    benchmark_task_list = []
    for index in unet_candidates:
        command = [
            "./benchmark_unet_candidate.sh",
            f"{candidate_trackers[index].unet_candidate_path}",
        ]
        benchmark_task_list.append(
            TaskTuple(
                args,
                command,
                check=False,
                command_need_device_id=True,
                cooling_time=10,
                result_need_device_id=True,
            )
        )
    benchmark_results = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=benchmark_task_list,
        function=run_command_wrapper,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )
    benchmark_results = sorted(benchmark_results, key=lambda tr: tr.device_id)
    grouped_benchmark_results = group_benchmark_results_by_device_id(benchmark_results)

    # Benchmarking baselines on each involved device
    worker_context_queue = create_worker_context_queue(args.devices)
    baseline_task_list = [
        TaskTuple(
            args,
            command=["./benchmark_unet_candidate.sh", str(unet_baseline_filepath)],
            check=False,
            command_need_device_id=True,
            result_need_device_id=True,
        )
    ] * len(grouped_benchmark_results)
    baseline_results = multiprocess_progress_wrapper(
        num_worker=len(args.devices),
        task_list=baseline_task_list,
        function=run_command_wrapper,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )
    baseline_results = sorted(baseline_results, key=lambda tr: tr.device_id)

    # Insert baseline results to the head of each list
    grouped_benchmark_results = [
        [x] + y for x, y in zip(baseline_results, grouped_benchmark_results)
    ]

    # Update candidate_tracker and extract strings which will be stored in unet_result_log
    dump_list = parse_grouped_benchmark_results(
        grouped_benchmark_results, candidate_trackers
    )

    with unet_result_log.open("w") as log_file:
        for dump_str in dump_list:
            log_file.write(dump_str)

    return unet_result_log


def autotune() -> None:
    args = parse_arguments()

    base_dir = Path(f"tuning_{datetime.now().strftime('%Y_%m_%d_%H_%M')}")
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate_trackers = []
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    log_file_path = setup_logging(args, log_dir=base_dir)
    print(log_file_path, end="\n\n")

    print("Generating candidates...")
    candidates, candidates_dir = generate_candidates(args, base_dir, candidate_trackers)
    print(f"Generated [{len(candidates)}] candidates in {candidates_dir}\n")
    if stop_after_phase == ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_files, compiled_dir = compile_candidates(
        args, base_dir, candidates, candidates_dir, candidate_trackers
    )
    print(f"Compiled [{len(compiled_files)}] files in {compiled_dir}\n")
    if stop_after_phase == ExecutionPhases.compile_candidates:
        return

    print("Benchmarking compiled candidates...")
    best_log = benchmark_compiled_candidates(
        args, base_dir, candidates_dir, compiled_files, candidate_trackers
    )
    print(f"Top candidates results are stored in {best_log}\n")
    if stop_after_phase == ExecutionPhases.benchmark_candidates:
        return

    print(f"Compiling top unet candidates...")
    unet_candidates = compile_unet_candidates(
        args, base_dir, best_log, candidate_trackers
    )
    print(f"Unet candidates compiled in {base_dir}\n")
    if stop_after_phase == ExecutionPhases.compile_unet_candidates:
        return

    print("Benchmarking unet candidates...")
    unet_result_log = benchmark_unet(
        args, base_dir, unet_candidates, candidate_trackers
    )
    print(f"Done, stored unet result in {unet_result_log}\n")
    if stop_after_phase == ExecutionPhases.benchmark_unet_candidates:
        return

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


def main():
    autotune()


if __name__ == "__main__":
    main()
