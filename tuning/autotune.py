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
from dataclasses import dataclass, field
from typing import Type, Optional, Callable, Iterable, Any
import pickle
from itertools import groupby
import random

"""
Sample Usage:

python autotune.py winograd 1286.mlir --lhs-dims=bmk --rhs-dims=bkn --tile-dims=*mnk --devices=1,3,5 --num-candidates=64


Recommended Trial Run:

python autotune.py winograd 1286.mlir --num-candidates=1


Dry Run Test (no gpu requried):

python autotune.py winograd 1286.mlir --num-candidates=64 --num-unet-candidates=10 --dry-run

"""


# Default values for num_candidates and devices, change it as needed
DEFAULT_NUM_CANDIDATES = 2048
DEFAULT_DEVICE_LIST = [0]

# Default values for max number of workers
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count() // 2
)  # the actual amount of worker that will be generated = max(min(max_cpu_workers//2, len(task_list)), 1)
"""note: Do not use all CPU cores"""

# Declare global variables at the module level for multiprocessing
worker_id = None
device_id = None
"""Do not need to change"""


@dataclass
class CandidateTracker:
    candidate_id: int
    mlir_path: Optional[Path] = None
    mlir_config_path: Optional[Path] = None
    configuration: Optional[tune.Configuration] = None
    compilation_successful: Optional[bool] = None
    compiled_vmfb_path: Optional[Path] = None
    compiled_vmfb_hash: Optional[str] = None
    first_benchmark_time: Optional[float] = None
    first_benchmark_device_id: Optional[int] = None
    unet_candidate_path: Optional[Path] = None
    unet_vmfb_hash: Optional[str] = None
    unet_benchmark_time: Optional[float] = None
    unet_benchmark_device_id: Optional[int] = None
    baseline_benchmark_time: Optional[float] = None
    calibrated_benchmark_diff: Optional[float] = None


@dataclass(frozen=True)
class PathConfig:
    # Preset constants
    global_config_prolog_mlir: Path = Path("./config_prolog.mlir")
    global_config_epilog_mlir: Path = Path("./config_epilog.mlir")
    compile_candidate_sh: Path = Path("./compile_candidate.sh")
    benchmark_dispatch_sh: Path = Path("./benchmark_dispatch.sh")
    compile_unet_candidate_sh: Path = Path("./compile_unet_candidate.sh")
    benchmark_unet_candidate_sh: Path = Path("./benchmark_unet_candidate.sh")
    unet_baseline_vmfb: Path = Path("./unet_baseline.vmfb")

    # Dynamic paths
    base_dir: Path = field(init=False)
    local_config_prolog_mlir: Path = field(init=False)
    local_config_epilog_mlir: Path = field(init=False)
    template_mlir: Path = field(init=False)
    candidates_dir: Path = field(init=False)
    candidate_configs_pkl: Path = field(init=False)
    compiled_dir: Path = field(init=False)
    compilefailed_dir: Path = field(init=False)
    candidate_vmfbs_txt: Path = field(init=False)
    dispatch_benchmark_result_log: Path = field(init=False)
    dispatch_benchmark_top_result_log: Path = field(init=False)

    unet_benchmark_result_log: Path = field(init=False)
    unet_result_log: Path = field(init=False)
    candidate_trackers_pkl: Path = field(init=False)

    # To be set outside of class
    log_file_path: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, "base_dir", self._name_base_dir())
        object.__setattr__(
            self, "local_config_prolog_mlir", self.base_dir / "config_prolog.mlir"
        )
        object.__setattr__(
            self, "local_config_epilog_mlir", self.base_dir / "config_epilog.mlir"
        )
        object.__setattr__(self, "template_mlir", self.base_dir / "template.mlir")
        object.__setattr__(self, "candidates_dir", self.base_dir / "candidates")
        object.__setattr__(
            self, "candidate_configs_pkl", self.candidates_dir / "configs.pkl"
        )
        object.__setattr__(self, "compiled_dir", self.candidates_dir / "compiled")
        object.__setattr__(self, "compilefailed_dir", self.candidates_dir / "failed")
        object.__setattr__(
            self, "candidate_vmfbs_txt", self.base_dir / "candidate_vmfbs.txt"
        )
        object.__setattr__(
            self, "dispatch_benchmark_result_log", self.base_dir / "results.log"
        )
        object.__setattr__(
            self, "dispatch_benchmark_top_result_log", self.base_dir / "best.log"
        )
        object.__setattr__(
            self, "unet_benchmark_result_log", self.base_dir / "unet_results.log"
        )
        object.__setattr__(self, "unet_result_log", self.base_dir / "unet_result.log")
        object.__setattr__(
            self, "candidate_trackers_pkl", self.base_dir / "candidate_trackers.pkl"
        )

    def _name_base_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        base_dir = Path(f"./tuning_{timestamp}")
        return base_dir

    def _set_log_file_path(self, log_file_path: Path):
        object.__setattr__(self, "log_file_path", log_file_path)

    def get_candidate_mlir_path(self, candidate_id: int) -> Path:
        return self.candidates_dir / f"{candidate_id}.mlir"

    def get_candidate_spec_mlir_path(self, candidate_id: int) -> Path:
        return self.candidates_dir / "configs" / f"{candidate_id}_spec.mlir"

    def get_exe_format(self, path: Path) -> str:
        return f"./{path.as_posix()}"


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
    device_id: Optional[int] = None


@dataclass
class ParsedDisptachBenchmarkResult:
    benchmark_time_in_seconds: float
    candidate_mlir: Path
    candidate_spec_mlir: Path


@dataclass
class DispatchBenchmarkResult:
    result_str: Optional[str] = None

    def get_tokens(self) -> list[str]:
        # e.g. ['0', 'Mean', 'Time:', '694.0']
        if self.result_str is None:
            return []
        try:
            return self.result_str.split()
        except:
            return []

    def get_candidate_id(self) -> Optional[int]:
        if len(self.get_tokens()) < 1:
            return None
        try:
            return int(self.get_tokens()[0])
        except ValueError:
            return None

    def get_benchmark_time(self) -> Optional[float]:
        if len(self.get_tokens()) < 4:
            return None
        try:
            return float(self.get_tokens()[3])
        except ValueError:
            return None

    def generate_sample_result(
        self, candidate_id: int = 0, mean_time: float = random.uniform(100.0, 500.0)
    ) -> str:
        # time unit is implicit and dependent on the output of iree-benchmark-module
        return f"{candidate_id}\tMean Time: {mean_time:.1f}\n"


@dataclass
class UnetBenchmarkResult:
    result_str: Optional[str] = None

    def get_tokens(self) -> list[str]:
        # e.g. ['Benchmarking:', '/sdxl-scripts/tuning/tuning_2024_07_19_08_55/unet_candidate_12.vmfb', 'on', 'device', '4', 'BM_main/process_time/real_time_median', '65.3', 'ms', '66.7', 'ms', '5', 'items_per_second=15.3201/s']
        if self.result_str is None:
            return []
        try:
            return self.result_str.split()
        except:
            return []

    def get_unet_candidate_path(self) -> Optional[str]:
        if len(self.get_tokens()) < 2:
            return None
        return self.get_tokens()[1]

    def get_candidate_id(self) -> Optional[int]:
        if self.get_unet_candidate_path():
            try:
                path_str = self.get_unet_candidate_path()
                return int(path_str.split("_")[-1].split(".")[0]) if path_str else None
            except ValueError:
                return None
        return None

    def get_device_id(self) -> Optional[int]:
        if len(self.get_tokens()) < 5:
            return None
        try:
            return int(self.get_tokens()[4])
        except ValueError:
            return None

    def get_benchmark_time(self) -> Optional[int | float]:
        if len(self.get_tokens()) < 7:
            return None
        try:
            return float(self.get_tokens()[6])
        except ValueError:
            return None

    def get_calibrated_result_str(self, change: float) -> Optional[str]:
        if self.result_str is None:
            return self.result_str

        benchmark_time = self.get_benchmark_time()
        if benchmark_time is None:
            return self.result_str

        # Format the change to be added to the string
        percentage_change = change * 100
        change_str = f"({percentage_change:+.3f}%)"

        # Use regex to find and replace the old benchmark time with the new one
        new_result_str = re.sub(
            r"(\d+(\.\d+)?)\s*ms",
            lambda m: f"{self.get_benchmark_time()} ms {change_str}",
            self.result_str,
            count=1,
        )

        return new_result_str

    def generate_sample_result(
        self,
        candidate_vmfb_path_str: str = "unet_baseline.vmfb",
        device_id: int = 0,
        t1: float = random.uniform(100.0, 500.0),  # time in ms
    ) -> str:
        return f"Benchmarking: {candidate_vmfb_path_str} on device {device_id}\nBM_run_forward/process_time/real_time_median\t    {t1:.3g} ms\t    {(t1+1):.3g} ms\t      5 items_per_second={t1/200:5f}/s\n\n"


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not attempt to run any modules or initialize the IREE runtime",
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
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
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


def setup_logging(args: argparse.Namespace, path_config: PathConfig):
    log_file_name = f"autotune_{args.mode}_{args.input_file.stem}.log"
    log_file_path = path_config.base_dir / log_file_name
    path_config._set_log_file_path(log_file_path)

    # Create file handler for logging to a file
    if path_config.log_file_path is None:
        raise
    file_handler = logging.FileHandler(path_config.log_file_path)
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


def handle_error(
    condition: bool,
    msg: str,
    level: int = logging.ERROR,
    error_type: Type[BaseException] = Exception,
    exit_program: bool = False,
) -> None:
    """If meets the condition, handles errors with logging and optional program exit"""
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
) -> Optional[subprocess.CompletedProcess]:
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

    res = run_command(task_tuple.args, task_tuple.command, task_tuple.check)
    if res is None:
        raise

    task_result = TaskResult(res)
    task_result.device_id = device_id if task_tuple.result_need_device_id else None

    time.sleep(task_tuple.cooling_time)

    return task_result


def multiprocess_progress_wrapper(
    num_worker: int,
    task_list: list,
    function: Callable,
    initializer: Optional[Callable] = None,
    initializer_inputs: Optional[Iterable[Any]] = None,
) -> list[Any]:
    """Wrapper of multiprocessing pool and progress bar"""
    results = []
    initializer_inputs = initializer_inputs or ()

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

    return results


def numerical_sort_key(path: Path) -> tuple[int | float, str]:
    """
    Define a sort key function that splits the filename into a numeric and a string part.
    Order: 0 | 0_a | 0_b | 1 | 1_a | 2
    """
    numeric_part: int | float
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
    hash_count: dict[str, list[int]] = {}

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
    args: argparse.Namespace,
    path_config: PathConfig,
    candidate_trackers: list[CandidateTracker],
) -> list[int]:
    """Generate candidate files for tuning. Returns the list of candidate indexes"""
    logging.info("generate_candidates()")

    try:
        shutil.copy(
            path_config.global_config_epilog_mlir, path_config.local_config_epilog_mlir
        )
        shutil.copy(
            path_config.global_config_prolog_mlir, path_config.local_config_prolog_mlir
        )
    except FileNotFoundError as e:
        handle_error(
            condition=True,
            msg=f"Configuration file not found: {e}",
            error_type=FileNotFoundError,
        )

    shutil.copy(args.input_file, path_config.template_mlir)

    mlirs = []
    try:
        logging.debug("Captured messages from tune.py:")
        tune.tune(
            input=str(path_config.template_mlir),
            output=str(path_config.candidates_dir),
            limit=args.num_candidates,
            num_subgroups=args.num_subgroups,
            lhs_dims=args.lhs_dims,
            rhs_dims=args.rhs_dims,
            tile_dims=args.tile_dims,
        )
        mlirs = sorted(
            path_config.candidates_dir.glob("*.mlir"), key=numerical_sort_key
        )
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

    candidate_configs = load_pickle(path_config.candidate_configs_pkl)
    candidate_configs.insert(0, None)  # No Configuration class for 0.mlir

    # Create candidate trackers
    assert len(mlirs) // 2 + 1 == len(candidate_configs)
    candidates = []
    for mlir in mlirs:
        if "_config.mlir" not in mlir.name:
            candidates.append(int(mlir.stem))
            new_candidate = CandidateTracker(
                candidate_id=int(mlir.stem),
                mlir_path=mlir,
                configuration=candidate_configs[int(mlir.stem)],
            )
            candidate_trackers.append(new_candidate)
        else:
            candidate_trackers[int(mlir.stem.split("_config")[0])].mlir_config_path = (
                mlir
            )

    handle_error(
        condition=(len(candidates) == 0), msg="Failed to generate any candidates"
    )

    return candidates


def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, list[int]]:
    """If a collision is found, generate a list of new indexes. If no collision, `unique_indexes = []`"""
    # Check if candidate produces tbe same .vmfb
    collision_detected, hash_list = find_collisions(index_hash_list)
    unique_indexes: list[int] = []
    if not collision_detected:
        return collision_detected, unique_indexes

    # If a collision is detected, select the first one from the collided list
    logging.warning("Collisions detected")
    for hash_val, indices in hash_list:
        if len(indices) != 1:
            logging.warning(f"Hash value '{hash_val}' collided at candidate {indices}.")
        unique_indexes.append(indices[0])

    return collision_detected, unique_indexes


def compile_candidates(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidates: list[int],
    candidate_trackers: list[CandidateTracker],
) -> list[int]:
    """Compile candidate files for tuning and record in candidate_vmfbs.txt. Returns the list of compiled candidate indexes."""
    logging.info("compile_candidates()")

    task_list = []
    for candidate_index in candidates:
        mlir_path = candidate_trackers[candidate_index].mlir_path
        assert mlir_path is not None
        command = [
            path_config.get_exe_format(path_config.compile_candidate_sh),
            args.mode,
            mlir_path.as_posix(),
        ]
        task_list.append(TaskTuple(args, command, check=False))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    compiled_files = sorted(
        path_config.compiled_dir.glob("*.vmfb"), key=numerical_sort_key
    )
    failed_files = sorted(
        path_config.compilefailed_dir.glob("*.mlir"), key=numerical_sort_key
    )

    total, good, bad = len(task_list), len(compiled_files), len(failed_files)
    compiling_rate = good / total * 100
    logging.critical(
        f"Total: {total} | Compiled: {good} | Failed: {bad} | Compiling Rate: {compiling_rate:.1f}%"
    )

    # Write compiled files to candidate_vmfbs.txt
    with path_config.candidate_vmfbs_txt.open("w") as f:
        for compiled_file in compiled_files:
            f.write(f"{compiled_file}\n")

    # Update candidate tracker
    for failed_file in failed_files:
        index = int(failed_file.stem)
        candidate_trackers[index].compilation_successful = False
    compiled_candidates = []
    compiled_candidates_hash_list = []
    for compiled_file in compiled_files:
        index = int(compiled_file.stem)
        compiled_candidates.append(index)
        candidate_trackers[index].compilation_successful = True
        candidate_trackers[index].compiled_vmfb_path = compiled_file
        compiled_vmfb_path = candidate_trackers[index].compiled_vmfb_path
        assert compiled_vmfb_path is not None
        hash_val = calculate_md5(compiled_vmfb_path)
        candidate_trackers[index].compiled_vmfb_hash = hash_val
        compiled_candidates_hash_list.append((index, hash_val))

    handle_error(
        condition=(good == 0), msg="Failed to compile all candidate .mlir files"
    )
    handle_error(
        condition=(compiling_rate < 10),
        msg=f"Compiling rate [{compiling_rate:.1f}%] < 10%",
        level=logging.WARNING,
    )

    collision_detected, unique_indexes = collision_handler(
        compiled_candidates_hash_list
    )
    if collision_detected:
        logging.critical(f"Remains [{len(unique_indexes)}] unique candidate indexes")

    return compiled_candidates if not collision_detected else unique_indexes


def parse_dispatch_benchmark_results(
    path_config: PathConfig,
    benchmark_results: list[TaskResult],
    candidate_trackers: list[CandidateTracker],
) -> tuple[list[ParsedDisptachBenchmarkResult], list[str]]:
    benchmark_result_configs = []
    dump_list = []

    for benchmark_result in benchmark_results:
        res_str = benchmark_result.result.stdout
        if res_str is None:
            continue
        res = DispatchBenchmarkResult(res_str)
        candidate_id = res.get_candidate_id()
        benchmark_time = res.get_benchmark_time()
        assert candidate_id is not None and benchmark_time is not None
        candidate_trackers[candidate_id].first_benchmark_time = benchmark_time
        dump_list.append(res_str)

        benchmark_result_configs.append(
            (
                ParsedDisptachBenchmarkResult(
                    benchmark_time,
                    path_config.get_candidate_mlir_path(candidate_id),
                    path_config.get_candidate_spec_mlir_path(candidate_id),
                )
            )
        )
    return benchmark_result_configs, dump_list


def generate_dryrun_dispatch_benchmark_results(
    compiled_candidates: list[int],
) -> list[TaskResult]:
    task_results = []
    for candidate_id in compiled_candidates:
        task_result = subprocess.CompletedProcess(
            args=[""],
            returncode=0,
            stdout=DispatchBenchmarkResult().generate_sample_result(
                candidate_id, mean_time=random.uniform(100.0, 500.0)
            ),
            stderr="",
        )
        task_results.append(TaskResult(task_result))
    return task_results


def benchmark_compiled_candidates(
    args: argparse.Namespace,
    path_config: PathConfig,
    compiled_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
):
    """Benchmark the candidate files and store the topN results in file (best.log)."""
    logging.info("benchmark_top_candidates()")

    if args.dry_run:
        benchmark_results = generate_dryrun_dispatch_benchmark_results(
            compiled_candidates
        )
    else:
        # Benchmarking dispatch candidates
        task_list = []
        for index in compiled_candidates:
            compiled_vmfb_path = candidate_trackers[index].compiled_vmfb_path
            assert compiled_vmfb_path is not None
            command = [
                path_config.get_exe_format(path_config.benchmark_dispatch_sh),
                compiled_vmfb_path.as_posix(),
            ]
            task_list.append(
                TaskTuple(args, command, check=False, command_need_device_id=True)
            )

        worker_context_queue = create_worker_context_queue(args.devices)
        benchmark_results = multiprocess_progress_wrapper(
            num_worker=len(args.devices),
            task_list=task_list,
            function=run_command_wrapper,
            initializer=init_worker_context,
            initializer_inputs=(worker_context_queue,),
        )

    (
        parsed_benchmark_results,
        dispatch_benchmark_dump_list,
    ) = parse_dispatch_benchmark_results(
        path_config, benchmark_results, candidate_trackers
    )

    with path_config.dispatch_benchmark_result_log.open("w") as log_file:
        log_file.writelines(dispatch_benchmark_dump_list)

    benchmarking_rate = (len(parsed_benchmark_results) / len(benchmark_results)) * 100
    logging.critical(
        f"Total: {len(benchmark_results)} | Benchmarked: {len(parsed_benchmark_results)} | Failed: {len(benchmark_results) - len(parsed_benchmark_results)} | Benchmarking Rate: {benchmarking_rate:.1f}%"
    )
    handle_error(
        condition=(len(benchmark_results) == 0),
        msg="Failed to benchmark all candidate .vmfb files",
    )

    # Select top candidates
    best_results = sorted(
        parsed_benchmark_results, key=lambda x: float(x.benchmark_time_in_seconds)
    )[: args.num_unet_candidates]
    logging.critical(f"Selected top[{len(best_results)}]")

    with path_config.dispatch_benchmark_top_result_log.open("w") as log_file:
        for result in best_results:
            log_file.write(
                f"{result.benchmark_time_in_seconds}\t{result.candidate_mlir.as_posix()}\t{result.candidate_spec_mlir.as_posix()}\n"
            )


def compile_unet_candidates(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidate_trackers: list[CandidateTracker],
) -> list[int]:
    """Compile U-Net candidates stored in best.log. Return the list of U-Net candidate files."""
    logging.info("compile_unet_candidates()")

    task_list = []
    with path_config.dispatch_benchmark_top_result_log.open("r") as log_file:
        for line in log_file:
            if "/0.mlir" not in line:
                input_file = line.strip().split()[2]
                command = [
                    path_config.get_exe_format(path_config.compile_unet_candidate_sh),
                    args.mode,
                    input_file,
                ]
                task_list.append(TaskTuple(args, command))

    num_worker = max(min(args.max_cpu_workers, len(task_list)), 1)  # at least 1 worker
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    unet_candidates_files = list(path_config.base_dir.glob("*.vmfb"))

    unet_candidates_indexes = []
    unet_candidates_hash_list = []

    # Update candidate tracker
    for unet_candidate in unet_candidates_files:
        index = int(unet_candidate.stem.split("_")[-1])
        candidate_trackers[index].unet_candidate_path = unet_candidate
        unet_candidate_path = candidate_trackers[index].unet_candidate_path
        assert unet_candidate_path is not None
        hash_val = calculate_md5(unet_candidate_path)
        candidate_trackers[index].unet_vmfb_hash = hash_val
        unet_candidates_hash_list.append((index, hash_val))
        unet_candidates_indexes.append(index)

    # Check if unet candidate produces tbe same .vmfb
    collision_detected, unique_unet_candidates_indexes = collision_handler(
        unet_candidates_hash_list
    )

    if collision_detected:
        logging.critical(
            f"Remains [{len(unique_unet_candidates_indexes)}] unique candidate indexes"
        )

    return (
        unique_unet_candidates_indexes
        if collision_detected
        else unet_candidates_indexes
    )


def sort_candidates_by_first_benchmark_times(
    candidate_indexes: list[int], candidate_trackers: list[CandidateTracker]
) -> list[int]:
    """Sorts candidate indexes based on their first benchmark times in ascending order"""
    # Get the first benchmark times, defaulting to a large number if None
    first_benchmark_times = [
        candidate_trackers[index].first_benchmark_time or float("inf")
        for index in candidate_indexes
    ]
    combined = list(zip(candidate_indexes, first_benchmark_times))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    sorted_indexes, _ = zip(*combined_sorted)

    return list(sorted_indexes)


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
    grouped_results: dict[int, list[TaskResult]] = {}
    for result in benchmark_results:
        assert result.device_id is not None
        if result.device_id not in grouped_results:
            grouped_results[result.device_id] = []
        grouped_results[result.device_id].append(result)

    grouped_benchmark_results = [
        grouped_results[device_id] for device_id in sorted(grouped_results)
    ]

    return grouped_benchmark_results


def parse_grouped_benchmark_results(
    path_config: PathConfig,
    grouped_benchmark_results: list[list[TaskResult]],
    candidate_trackers: list[CandidateTracker],
) -> list[str]:
    """Update candidate_trackers and collect strings"""
    dump_list = []
    incomplete_list: list[tuple[int, Optional[int]]] = (
        []
    )  # format: [(candidate_id, device_id)], baseline will have candidate_id=0

    for same_device_results in grouped_benchmark_results:
        dump_unsort_list: list[tuple[float, str]] = []
        for unet_candidate_result in same_device_results:
            # Skip if benchmark failed.
            result_str = unet_candidate_result.result.stdout
            if result_str is None:
                continue

            res = UnetBenchmarkResult(result_str)
            device_id = res.get_device_id()

            # Record baseline benchmarking result.
            unet_candidate_path = res.get_unet_candidate_path()
            if (
                unet_candidate_path is not None
                and str(path_config.unet_baseline_vmfb) in unet_candidate_path
            ):
                baseline_time = res.get_benchmark_time()
                if baseline_time is None:
                    incomplete_list.append((0, device_id))
                    continue
                dump_list.append(result_str)
                continue

            # Record candidate benchmarking result.
            c_id = res.get_candidate_id()
            assert c_id is not None
            candidate_time = res.get_benchmark_time()
            if candidate_time is None:
                incomplete_list.append((c_id, device_id))
                continue
            candidate_trackers[c_id].unet_benchmark_time = candidate_time
            candidate_trackers[c_id].unet_benchmark_device_id = device_id
            # Skip improvement calculation if no baseline data.
            if baseline_time is None:
                dump_unsort_list.append((candidate_time, result_str))
                continue
            # Calculate candidate improvement based baseline.
            candidate_trackers[c_id].baseline_benchmark_time = baseline_time
            calibrated_benchmark_diff = (candidate_time - baseline_time) / baseline_time
            candidate_trackers[c_id].calibrated_benchmark_diff = (
                calibrated_benchmark_diff
            )
            dump_str = res.get_calibrated_result_str(calibrated_benchmark_diff)
            assert dump_str is not None
            dump_unsort_list.append((candidate_time, dump_str))

        # Sort unet candidate benchmarking result str in ascending time order.
        dump_list = dump_list + [
            dump_str for _, dump_str in sorted(dump_unsort_list, key=lambda x: x[0])
        ]

    # Store incomplete .vmfb file at the end of dump_list.
    for index, device_id in incomplete_list:
        index_to_path = lambda index: (
            f"{path_config.unet_baseline_vmfb.as_posix()}"
            if index == 0
            else f"{candidate_trackers[index].unet_candidate_path}"
        )
        error_msg = f"Benchmarking result of {index_to_path(index)} on deivce {device_id} is incomplete"
        handle_error(condition=True, msg=error_msg, level=logging.WARNING)
        dump_list.append(error_msg + "\n")

    return dump_list


def generate_dryrun_unet_benchmark_results(
    unet_vmfb_paths: list[Path],
) -> list[TaskResult]:
    task_results = []
    start = random.uniform(100.0, 500.0)
    device_id = 0
    for candidate_vmfb_path in unet_vmfb_paths:
        task_result = subprocess.CompletedProcess(
            args=[""],
            returncode=0,
            stdout=UnetBenchmarkResult().generate_sample_result(
                candidate_vmfb_path_str=candidate_vmfb_path.as_posix(),
                device_id=device_id,
                t1=start,
            ),
            stderr="",
        )
        start += random.uniform(-5.0, 8.0)
        task_results.append(TaskResult(task_result, device_id))
    return task_results


def dryrun_benchmark_unet(
    path_config: PathConfig,
    unet_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
):

    unet_vmfb_paths = [path_config.unet_baseline_vmfb]
    for i in unet_candidates:
        unet_candidate_path = candidate_trackers[i].unet_candidate_path
        assert unet_candidate_path is not None
        unet_vmfb_paths.append(unet_candidate_path)

    benchmark_results = generate_dryrun_unet_benchmark_results(unet_vmfb_paths)
    grouped_benchmark_results = group_benchmark_results_by_device_id(benchmark_results)

    # Update candidate_tracker and extract strings which will be stored in unet_result_log.
    dump_list = parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )

    with path_config.unet_result_log.open("w") as log_file:
        log_file.writelines(dump_list)


def benchmark_unet(
    args: argparse.Namespace,
    path_config: PathConfig,
    unet_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
):
    """Benchmark U-Net candidate files and log the results."""
    logging.info("benchmark_unet()")

    if args.dry_run:
        dryrun_benchmark_unet(path_config, unet_candidates, candidate_trackers)
        return

    # Benchmarking unet candidates
    worker_context_queue = create_worker_context_queue(args.devices)
    benchmark_task_list = []
    for index in unet_candidates:
        unet_candidate_path = candidate_trackers[index].unet_candidate_path
        assert unet_candidate_path is not None
        command = [
            path_config.get_exe_format(path_config.benchmark_unet_candidate_sh),
            unet_candidate_path.as_posix(),
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
    benchmark_results = sorted(benchmark_results, key=lambda br: br.device_id)
    grouped_benchmark_results = group_benchmark_results_by_device_id(benchmark_results)

    # Benchmarking baselines on each involved device
    worker_context_queue = create_worker_context_queue(args.devices)
    baseline_task_list = [
        TaskTuple(
            args,
            command=[
                path_config.get_exe_format(path_config.benchmark_unet_candidate_sh),
                path_config.unet_baseline_vmfb.as_posix(),
            ],
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
        path_config, grouped_benchmark_results, candidate_trackers
    )

    with path_config.unet_result_log.open("w") as log_file:
        log_file.writelines(dump_list)


def autotune(args: argparse.Namespace) -> None:
    path_config = PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)

    candidate_trackers: list[CandidateTracker] = []
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    setup_logging(args, path_config)
    print(path_config.log_file_path, end="\n\n")

    print("Generating candidates...")
    candidates = generate_candidates(args, path_config, candidate_trackers)
    print(f"Generated [{len(candidates)}] candidates in {path_config.candidates_dir}\n")
    if stop_after_phase == ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_candidates = compile_candidates(
        args, path_config, candidates, candidate_trackers
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == ExecutionPhases.compile_candidates:
        return

    print("Benchmarking compiled candidates...")
    benchmark_compiled_candidates(
        args, path_config, compiled_candidates, candidate_trackers
    )
    print(
        f"All dispatch benchmark results are stored in {path_config.dispatch_benchmark_result_log}"
    )
    print(
        f"Top candidates results are stored in {path_config.dispatch_benchmark_top_result_log}\n"
    )
    if stop_after_phase == ExecutionPhases.benchmark_candidates:
        return

    print(f"Compiling top unet candidates...")
    unet_candidates = compile_unet_candidates(args, path_config, candidate_trackers)
    print(f"Unet candidates compiled in {path_config.base_dir}\n")
    if stop_after_phase == ExecutionPhases.compile_unet_candidates:
        return

    print("Benchmarking unet candidates...")
    benchmark_unet(args, path_config, unet_candidates, candidate_trackers)
    print(f"Done, stored unet result in {path_config.unet_result_log}\n")
    if stop_after_phase == ExecutionPhases.benchmark_unet_candidates:
        return

    with open(path_config.candidate_trackers_pkl, "wb") as file:
        pickle.dump(candidate_trackers, file)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed log in:")
    print(path_config.log_file_path)

    for candidate in candidate_trackers:
        logging.debug(candidate)
        if args.verbose:
            print(candidate)


def main():
    autotune(parse_arguments())


if __name__ == "__main__":
    main()
