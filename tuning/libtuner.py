# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
import candidate_gen
from tqdm import tqdm
import re
import hashlib
from dataclasses import dataclass, field
from typing import Type, Optional, Callable, Iterable, Any
import pickle
import iree.runtime as ireert
import random
from abc import ABC, abstractmethod


# Default values for num_candidates and devices, change it as needed
DEFAULT_NUM_CANDIDATES = 2048
DEFAULT_DEVICE_LIST = ["hip://0"]

# Default values for max number of workers
DEFAULT_MAX_CPU_WORKERS = (
    multiprocessing.cpu_count() // 2
)  # the actual amount of worker that will be generated = max(min(max_cpu_workers//2, len(task_list)), 1)
"""note: Do not use all CPU cores"""

# Declare global variables at the module level for multiprocessing
worker_id = None
device_id = None
"""Do not need to change"""

# Declare special symbols for libtuner to search and locate
DEVICE_ID_RE = "DEVICE_ID_RE"


@dataclass
class CandidateTracker:
    candidate_id: int
    dispatch_mlir_path: Optional[Path] = None
    dispatch_config_path: Optional[Path] = None
    configuration: Optional[candidate_gen.Configuration] = None
    compilation_successful: Optional[bool] = None
    compiled_dispatch_path: Optional[Path] = None
    compiled_dispatch_hash: Optional[str] = None
    first_benchmark_time: Optional[float] = None
    first_benchmark_device_id: Optional[str] = None
    spec_path: Optional[Path] = None
    compiled_model_path: Optional[Path] = None
    compiled_model_hash: Optional[str] = None
    model_benchmark_time: Optional[float] = None
    model_benchmark_device_id: Optional[str] = None
    baseline_benchmark_time: Optional[float] = None
    calibrated_benchmark_diff: Optional[float] = None


@dataclass(frozen=True)
class PathConfig:
    # Preset constants
    global_config_prolog_mlir: Path = Path("./config_prolog.mlir")
    global_config_epilog_mlir: Path = Path("./config_epilog.mlir")
    model_baseline_vmfb: Path = Path("./baseline.vmfb")

    # Dynamic paths
    base_dir: Path = field(init=False)
    local_config_prolog_mlir: Path = field(init=False)
    local_config_epilog_mlir: Path = field(init=False)
    template_mlir: Path = field(init=False)
    candidates_dir: Path = field(init=False)
    candidate_configs_pkl: Path = field(init=False)
    compiled_dir: Path = field(init=False)
    compile_failed_dir: Path = field(init=False)
    spec_dir: Path = field(init=False)

    output_unilog: Path = field(init=False)
    result_summary_log: Path = field(init=False)
    candidate_trackers_pkl: Path = field(init=False)

    # To be set outside of class
    run_log: Optional[Path] = field(init=False, default=None)

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
        object.__setattr__(self, "compile_failed_dir", self.candidates_dir / "failed")
        object.__setattr__(self, "spec_dir", self.candidates_dir / "specs")
        object.__setattr__(self, "output_unilog", self.base_dir / "output.log")
        object.__setattr__(
            self, "result_summary_log", self.base_dir / "result_summary.log"
        )
        object.__setattr__(
            self, "candidate_trackers_pkl", self.base_dir / "candidate_trackers.pkl"
        )

    def _name_base_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        base_dir = Path(f"./tuning_{timestamp}")
        return base_dir

    def _set_run_log(self, run_log: Path):
        object.__setattr__(self, "run_log", run_log)

    def get_candidate_mlir_path(self, candidate_id: int) -> Path:
        return self.candidates_dir / f"{candidate_id}.mlir"

    def get_candidate_spec_mlir_path(self, candidate_id: int) -> Path:
        return self.candidates_dir / "specs" / f"{candidate_id}_spec.mlir"

    def get_exe_format(self, path: Path) -> str:
        return f"./{path.as_posix()}"

    def get_compiled_dispatch_index(self, file_path: Path) -> int:
        return int(file_path.stem)

    def get_candidate_spec_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}_spec.mlir"

    def get_compiled_model_index(self, file_path: Path) -> int:
        return int(file_path.stem.split("_")[-1])


class TuningClient(ABC):
    @abstractmethod
    def get_dispatch_compile_command(
        self, candidate_tracker: CandidateTracker
    ) -> list[str]:
        pass

    @abstractmethod
    def get_dispatch_benchmark_command(
        self, candidate_tracker: CandidateTracker
    ) -> list[str]:
        pass

    @abstractmethod
    def get_model_compile_command(
        self, candidate_tracker: CandidateTracker
    ) -> list[str]:
        pass

    @abstractmethod
    def get_model_benchmark_command(
        self, candidate_tracker: CandidateTracker
    ) -> list[str]:
        pass


@dataclass
class TaskPack:
    args: argparse.Namespace
    candidate_id: int
    command: list[str]
    check: bool = True
    command_need_device_id: bool = False
    cooling_time: int = 0


@dataclass
class TaskResult:
    result: subprocess.CompletedProcess
    candidate_id: int
    device_id: str


@dataclass
class ParsedDisptachBenchmarkResult:
    candidate_id: int
    benchmark_time_in_seconds: float
    candidate_mlir: Path
    candidate_spec_mlir: Path


@dataclass
class IREEBenchmarkResult:
    # Default format follows output of iree-benchmark-module
    candidate_id: int
    result_str: str

    def extract_key(self, pattern: str) -> Optional[str]:
        if not self.result_str:
            return None
        match = re.search(pattern, self.result_str)
        if not match:
            return None
        return match.group(1)

    def get_mean_time(self) -> Optional[float]:
        pattern = r"process_time/real_time_mean\s+([\d.]+)\s\w{2}"
        time_str = self.extract_key(pattern)
        if not time_str:
            return None
        try:
            return float(time_str)
        except ValueError:
            return None

    def get_median_time(self) -> Optional[float]:
        pattern = r"process_time/real_time_median\s+([\d.]+)\s\w{2}"
        time_str = self.extract_key(pattern)
        if not time_str:
            return None
        try:
            return float(time_str)
        except ValueError:
            return None


def generate_display_DBR(
    candidate_id: int = 0, mean_time: float = random.uniform(100.0, 500.0)
) -> str:
    """Generate dispatch_benchmark_result string for displaying"""
    # time unit is implicit and dependent on the output of iree-benchmark-module
    return f"{candidate_id}\tMean Time: {mean_time:.1f}\n"


def generate_display_MBR(
    candidate_vmfb_path_str: str = "baseline.vmfb",
    device_id: str = "0",
    t1: float = random.uniform(100.0, 500.0),
    calibrated_diff: Optional[float] = None,
) -> str:
    """Generate model_benchmark_result string for displaying"""
    # time unit is implicit and dependent on the output of iree-benchmark-module
    head_str = f"Benchmarking: {candidate_vmfb_path_str} on device {device_id}\n"
    res_str = f"process_time/real_time_median\t    {t1:.3g} ms\n\n"
    if calibrated_diff:
        percentage_change = calibrated_diff * 100
        change_str = f"({percentage_change:+.3f}%)"
        res_str = f"process_time/real_time_median\t    {t1:.3g} ms {change_str}\n\n"
    return head_str + res_str


def extract_driver_names(user_devices: list[str]) -> set[str]:
    """Extract driver names from the user devices"""
    return {device.split("://")[0] for device in user_devices}


def fetch_available_devices(drivers: list[str]) -> list[str]:
    """
    Extract all available devices on the user's machine for the provided drivers
    Only the user provided drivers will be queried
    """
    all_device_ids: list[str] = []

    for driver_name in drivers:
        try:
            driver = ireert.get_driver(driver_name)
            devices = driver.query_available_devices()
            all_device_ids.extend(
                f"{driver_name}://{device['path']}" for device in devices
            )
        except ValueError as e:
            handle_error(
                condition=True,
                msg=f"Could not initialize driver {driver_name}: {e}",
                error_type=ValueError,
                exit_program=True,
            )

    return all_device_ids


def parse_devices(devices_str: str) -> list[str]:
    """
    Parse a comma-separated list of device IDs e.g.:
    --devices=hip://0,local-sync://default -> ["hip://0", "local-sync://default"]).
    """
    devices = [device.strip() for device in devices_str.split(",")]
    for device in devices:
        if "://" not in device or not device:
            handle_error(
                condition=True,
                msg=f"Invalid device list: {devices_str}. Error: {ValueError()}",
                error_type=argparse.ArgumentTypeError,
            )
    return devices


def validate_devices(user_devices: list[str]) -> None:
    """Validates the user provided devices against the devices extracted by the IREE Runtime"""
    user_drivers = extract_driver_names(user_devices)

    available_devices = fetch_available_devices(list(user_drivers))

    for device in user_devices:
        handle_error(
            condition=(device not in available_devices),
            msg=f"Invalid device specified: {device}\nFetched available devices: {available_devices}",
            error_type=argparse.ArgumentError,
            exit_program=True,
        )


class ExecutionPhases(str, Enum):
    dont_stop = ""
    generate_candidates = "generate-candidates"
    compile_dispatches = "compile-dispatches"
    benchmark_dispatches = "benchmark-dispatches"
    compile_models = "compile-models"
    benchmark_models = "benchmark-models"


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
        help="Comma-separated list of device IDs (e.g., --devices=hip://,hip://GPU-UUID).",
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
        "--num-model-candidates",
        help="Maximum number of stage 2 candidates",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not attempt to run any modules or initialize the IREE runtime",
    )

    # candidate_gen.tune() options
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of candidates to be generated by candidate_gen.py (default: {DEFAULT_NUM_CANDIDATES})",
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
    run_log_path = path_config.base_dir / log_file_name
    path_config._set_run_log(run_log_path)

    # Create file handler for logging to a file
    if path_config.run_log is None:
        raise
    file_handler = logging.FileHandler(path_config.run_log)
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

    # config logger in candidate_gen.py
    tune_logger = logging.getLogger("tune")
    tune_logger.setLevel(logging.DEBUG)

    # Log all arguments
    logging.info(f"Input Arguments:")
    for arg, value in vars(args).items():
        tune_logger.info(f"{arg}: {value}")


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
            logging.info(f"stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
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


def run_command_wrapper(task_tuple: TaskPack) -> TaskResult:
    """pool.imap_unordered can't iterate an iterable of iterables input, this function helps dividing arguments"""
    if task_tuple.command_need_device_id:
        # worker searches for special symbol and substitute to correct device_id
        pattern = re.compile(re.escape(DEVICE_ID_RE))
        task_tuple.command = [
            pattern.sub(str(device_id), s) for s in task_tuple.command
        ]

    res = run_command(task_tuple.args, task_tuple.command, task_tuple.check)
    if res is None:
        raise

    task_result = TaskResult(
        res, task_tuple.candidate_id, device_id=str(-1)
    )  # main process
    if device_id:
        task_result = TaskResult(res, task_tuple.candidate_id, device_id)  # sub process

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


def save_pickle(file_path: Path, input_list: list[Any]) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(input_list, file)


def append_to_file(lines: list[str], filepath: Path, title: str = "") -> None:
    """Appends new content to the end of the output.log."""
    title_str = "=" * 5 + f" {title} " + "=" * 5 + "\n" if title != "" else ""
    with open(filepath, "a") as file:
        file.write(title_str)
        file.writelines(lines)
        file.write("\n")


def generate_candidates(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
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
        logging.debug("Captured messages from candidate_gen.py:")
        candidate_gen.tune(
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
        # Capture and log debug messages from candidate_gen.py
        tune_logger = logging.getLogger("tune")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                tune_logger.handlers.append(handler)
        tune_logger.exception("Error in candidate_gen.py:")
        raise
    logging.debug("candidate_gen.py ends")

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
                dispatch_mlir_path=mlir,
                configuration=candidate_configs[int(mlir.stem)],
            )
            candidate_trackers.append(new_candidate)
        else:
            candidate_trackers[
                int(mlir.stem.split("_config")[0])
            ].dispatch_config_path = mlir

    handle_error(
        condition=(len(candidates) == 0), msg="Failed to generate any candidates"
    )

    logging.critical(f"Generated [{len(candidates)}] candidates")

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


def compile_dispatches(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidates: list[int],
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
) -> list[int]:
    """Compile candidate files for tuning and record in candidate_vmfbs.txt. Returns the list of compiled candidate indexes."""
    logging.info("compile_candidates()")

    if not candidates:
        logging.info("No candidates to compile.")
        return []

    task_list = [
        TaskPack(
            args,
            candidate_id=i,
            command=tuning_client.get_dispatch_compile_command(candidate_trackers[i]),
            check=False,
        )
        for i in candidates
    ]
    num_worker = min(args.max_cpu_workers, len(task_list))
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    compiled_files = sorted(
        path_config.compiled_dir.glob("*.vmfb"), key=numerical_sort_key
    )
    failed_files = sorted(
        path_config.compile_failed_dir.glob("*.mlir"), key=numerical_sort_key
    )

    total, good, bad = len(task_list), len(compiled_files), len(failed_files)
    compiling_rate = good / total * 100
    logging.critical(
        f"Total: {total} | Compiled: {good} | Failed: {bad} | Compiling Rate: {compiling_rate:.1f}%"
    )

    # Update candidate tracker
    for failed_file in failed_files:
        index = path_config.get_compiled_dispatch_index(failed_file)
        candidate_trackers[index].compilation_successful = False
    compiled_candidates = []
    compiled_candidates_hash_list = []
    for compiled_file in compiled_files:
        index = path_config.get_compiled_dispatch_index(compiled_file)
        compiled_candidates.append(index)
        candidate_trackers[index].compilation_successful = True
        candidate_trackers[index].compiled_dispatch_path = compiled_file
        compiled_vmfb_path = candidate_trackers[index].compiled_dispatch_path
        assert compiled_vmfb_path is not None
        hash_val = calculate_md5(compiled_vmfb_path)
        candidate_trackers[index].compiled_dispatch_hash = hash_val
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
        candidate_id = benchmark_result.candidate_id
        if res_str is None:
            continue
        res = IREEBenchmarkResult(candidate_id, res_str)
        benchmark_time = res.get_mean_time()
        assert benchmark_time is not None
        candidate_trackers[candidate_id].first_benchmark_time = benchmark_time
        candidate_trackers[candidate_id].spec_path = (
            path_config.spec_dir / path_config.get_candidate_spec_filename(candidate_id)
        )
        mlir_path = candidate_trackers[candidate_id].dispatch_mlir_path
        spec_path = candidate_trackers[candidate_id].spec_path
        assert mlir_path is not None and spec_path is not None
        dump_list.append(generate_display_DBR(candidate_id, benchmark_time))

        benchmark_result_configs.append(
            (
                ParsedDisptachBenchmarkResult(
                    candidate_id,
                    benchmark_time,
                    mlir_path,
                    spec_path,
                )
            )
        )
    return benchmark_result_configs, dump_list


def generate_sample_task_result(
    stdout: str, candidate_id: int, device_id: str
) -> TaskResult:
    res = subprocess.CompletedProcess(
        args=[""],
        stdout=stdout,
        returncode=0,
    )
    return TaskResult(result=res, candidate_id=candidate_id, device_id=device_id)


def generate_dryrun_dispatch_benchmark_results(
    compiled_candidates: list[int],
) -> list[TaskResult]:
    logging.info("generate_dryrun_dispatch_benchmark_results")

    task_results = [
        generate_sample_task_result(
            f"process_time/real_time_mean    {random.uniform(100.0, 500.0):.3g} ms",
            i,
            str(0),
        )
        for i in compiled_candidates
    ]

    return task_results


def generate_dryrun_model_benchmark_results(
    model_candidates: list[int],
) -> tuple[list[TaskResult], list[TaskResult]]:
    candidate_results = []
    for i, j in enumerate(model_candidates):
        stdout = (
            f"process_time/real_time_median    {random.uniform(100.0, 500.0):.3g} ms"
        )
        candidate_results.append(generate_sample_task_result(stdout, j, str(i % 3)))

    baseline_results = [
        generate_sample_task_result(
            f"process_time/real_time_median    {random.uniform(100.0, 500.0):.3g} ms",
            0,
            str(i),
        )
        for i in range(3)
    ]

    return candidate_results, baseline_results


def benchmark_dispatches(
    args: argparse.Namespace,
    path_config: PathConfig,
    compiled_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
):
    """Benchmark the candidate files and store the topN results in file (best.log)."""
    logging.info("benchmark_top_candidates()")

    if args.dry_run:
        benchmark_results = generate_dryrun_dispatch_benchmark_results(
            compiled_candidates
        )
    else:
        # Benchmarking dispatch candidates
        task_list = [
            TaskPack(
                args,
                candidate_id=i,
                command=tuning_client.get_dispatch_benchmark_command(
                    candidate_trackers[i]
                ),
                check=False,
                command_need_device_id=True,
            )
            for i in compiled_candidates
        ]
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
    append_to_file(
        dispatch_benchmark_dump_list,
        filepath=path_config.output_unilog,
        title="All Dispatch Benchmark Results",
    )

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
    )[: args.num_model_candidates]
    logging.critical(f"Selected top[{len(best_results)}]")

    dump_list = [
        f"{result.benchmark_time_in_seconds}\t{result.candidate_mlir.as_posix()}\t{result.candidate_spec_mlir.as_posix()}\n"
        for result in best_results
    ]
    append_to_file(
        dump_list, filepath=path_config.output_unilog, title="Top Candidates Results"
    )

    top_candidates = [result.candidate_id for result in best_results]
    return top_candidates


def compile_models(
    args: argparse.Namespace,
    path_config: PathConfig,
    candidates: list[int],
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
) -> list[int]:
    """Compile U-Net candidates stored in best.log. Return the list of U-Net candidate files."""
    logging.info("compile_models()")

    candidate_trackers[0].compiled_model_path = path_config.model_baseline_vmfb

    if args.dry_run:
        for i in candidates:
            candidate_trackers[i].compiled_model_path = Path(f"model_{i}.vmfb")
        return candidates

    if not candidates:
        logging.info("No model candidates to compile.")
        return []

    task_list = [
        TaskPack(
            args,
            candidate_id=i,
            command=tuning_client.get_model_compile_command(candidate_trackers[i]),
        )
        for i in candidates
        if i != 0
    ]
    num_worker = min(args.max_cpu_workers, len(task_list))
    multiprocess_progress_wrapper(
        num_worker=num_worker, task_list=task_list, function=run_command_wrapper
    )

    model_candidates_files = list(path_config.base_dir.glob("*.vmfb"))

    model_candidates_indexes = []
    model_candidates_hash_list = []

    # Update candidate tracker
    for model_candidate in model_candidates_files:
        assert model_candidate is not None
        index = path_config.get_compiled_model_index(model_candidate)
        candidate_trackers[index].compiled_model_path = model_candidate
        hash_val = calculate_md5(model_candidate)
        candidate_trackers[index].compiled_model_hash = hash_val
        model_candidates_hash_list.append((index, hash_val))
        model_candidates_indexes.append(index)

    # Check if model candidate produces tbe same .vmfb
    collision_detected, unique_model_candidates_indexes = collision_handler(
        model_candidates_hash_list
    )

    if collision_detected:
        logging.critical(
            f"Remains [{len(unique_model_candidates_indexes)}] unique candidate indexes"
        )

    return (
        unique_model_candidates_indexes
        if collision_detected
        else model_candidates_indexes
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
    grouped_results: dict[str, list[TaskResult]] = {}
    for result in benchmark_results:
        assert result.device_id is not None
        if result.device_id not in grouped_results:
            grouped_results[result.device_id] = []
        grouped_results[result.device_id].append(result)

    grouped_benchmark_results = [
        grouped_results[device_id] for device_id in sorted(grouped_results)
    ]

    return grouped_benchmark_results


def parse_model_benchmark_results(
    candidate_trackers: list[CandidateTracker],
    candidate_results: list[TaskResult],
    baseline_results: list[TaskResult],
):
    candidate_results = sorted(candidate_results, key=lambda br: br.device_id)
    baseline_results = sorted(baseline_results, key=lambda tr: tr.device_id)

    # Assign candidates to same groups by device_id
    grouped_candidate_results = group_benchmark_results_by_device_id(candidate_results)

    # Insert baseline results to the head of each list
    grouped_benchmark_results = [
        [x] + y for x, y in zip(baseline_results, grouped_candidate_results)
    ]

    dump_list = []
    incomplete_list: list[tuple[int, Optional[str]]] = (
        []
    )  # format: [(candidate_id, device_id)], baseline will have candidate_id=0

    for same_device_results in grouped_benchmark_results:
        dump_unsort_list: list[tuple[float, str]] = []
        for task_result in same_device_results:
            # Skip if benchmark failed.
            result_str = task_result.result.stdout
            candidate_id = task_result.candidate_id
            device_id = task_result.device_id

            if result_str is None:
                # TODO: change incomplete process detection
                continue

            res = IREEBenchmarkResult(candidate_id, result_str)
            benchmark_time = res.get_median_time()

            # Check completion
            if benchmark_time == None:
                incomplete_list.append((candidate_id, device_id))
                continue
            assert benchmark_time is not None

            # Record baseline benchmarking result and skip rest processes
            if candidate_id == 0:
                baseline_time = benchmark_time
                baseline_vmfb_path = candidate_trackers[
                    candidate_id
                ].compiled_model_path
                assert baseline_vmfb_path is not None
                dump_str = generate_display_MBR(
                    candidate_vmfb_path_str=baseline_vmfb_path.as_posix(),
                    device_id=device_id,
                    t1=benchmark_time,
                )
                dump_list.append(dump_str)
                continue

            # Update candidate_tracker
            candidate_trackers[candidate_id].model_benchmark_time = benchmark_time
            candidate_trackers[candidate_id].model_benchmark_device_id = device_id

            # Skip improvement calculation if no baseline data.
            if baseline_time is None:
                dump_unsort_list.append((benchmark_time, result_str))
                continue

            # Calculate candidate improvement based on baseline.
            candidate_trackers[candidate_id].baseline_benchmark_time = baseline_time
            calibrated_benchmark_diff = (benchmark_time - baseline_time) / baseline_time
            candidate_trackers[candidate_id].calibrated_benchmark_diff = (
                calibrated_benchmark_diff
            )

            # Collect candidate dump str
            candidate_vmfb_path = candidate_trackers[candidate_id].compiled_model_path
            assert candidate_vmfb_path is not None
            dump_str = generate_display_MBR(
                candidate_vmfb_path_str=candidate_vmfb_path.as_posix(),
                device_id=device_id,
                t1=benchmark_time,
                calibrated_diff=calibrated_benchmark_diff,
            )

            dump_unsort_list.append((benchmark_time, dump_str))

        # Sort model candidate benchmarking result str in ascending time order.
        dump_list = dump_list + [
            dump_str for _, dump_str in sorted(dump_unsort_list, key=lambda x: x[0])
        ]

    # Store incomplete .vmfb file at the end of dump_list.
    for index, device in incomplete_list:
        file_path = candidate_trackers[index].compiled_model_path
        assert file_path is not None
        error_msg = f"Benchmarking result of {file_path.as_posix()} on deivce {device} is incomplete"
        handle_error(condition=True, msg=error_msg, level=logging.WARNING)
        dump_list.append(error_msg + "\n")

    return dump_list


def benchmark_models(
    args: argparse.Namespace,
    path_config: PathConfig,
    model_candidates: list[int],
    candidate_trackers: list[CandidateTracker],
    tuning_client: TuningClient,
):
    """Benchmark U-Net candidate files and log the results."""
    logging.info("benchmark_models()")

    if args.dry_run:
        candidate_results, baseline_results = generate_dryrun_model_benchmark_results(
            model_candidates
        )
    else:
        # Benchmarking model candidates
        worker_context_queue = create_worker_context_queue(args.devices)
        benchmark_task_list = [
            TaskPack(
                args,
                candidate_id=i,
                command=tuning_client.get_model_benchmark_command(
                    candidate_trackers[i]
                ),
                check=False,
                command_need_device_id=True,
                cooling_time=10,
            )
            for i in model_candidates
        ]
        candidate_results = multiprocess_progress_wrapper(
            num_worker=len(args.devices),
            task_list=benchmark_task_list,
            function=run_command_wrapper,
            initializer=init_worker_context,
            initializer_inputs=(worker_context_queue,),
        )

        # Benchmarking baselines on each involved device
        candidate_trackers[0].compiled_model_path = path_config.model_baseline_vmfb
        worker_context_queue = create_worker_context_queue(args.devices)
        baseline_task_list = [
            TaskPack(
                args,
                candidate_id=0,
                command=tuning_client.get_model_benchmark_command(
                    candidate_trackers[0]
                ),
                check=False,
                command_need_device_id=True,
            )
        ] * len(group_benchmark_results_by_device_id(candidate_results))
        baseline_results = multiprocess_progress_wrapper(
            num_worker=len(args.devices),
            task_list=baseline_task_list,
            function=run_command_wrapper,
            initializer=init_worker_context,
            initializer_inputs=(worker_context_queue,),
        )

    dump_list = parse_model_benchmark_results(
        candidate_trackers, candidate_results, baseline_results
    )

    append_to_file(
        dump_list, filepath=path_config.output_unilog, title="Model Benchmark Results"
    )


def summerize_top_candidates(
    path_config: PathConfig, candidate_trackers: list[CandidateTracker]
):
    dump_list = []
    top_candidates = []
    for candidate in candidate_trackers:
        if candidate.candidate_id == 0 or candidate.model_benchmark_time is None:
            continue
        top_candidates.append(
            (candidate.candidate_id, candidate.model_benchmark_time)
        )  # collect (id, time)

    top_candidates = sorted(
        top_candidates, key=lambda x: x[1]
    )  # sort the list in ascending benchmark time order
    top_candidate_ids = [item[0] for item in top_candidates]  # get list of candidate id

    for candidate_id in top_candidate_ids:
        candidate = candidate_trackers[candidate_id]
        assert candidate.dispatch_config_path is not None
        with open(candidate.dispatch_config_path, "r") as file:
            config_file_contents = file.read()
        final_str = f"Candidate {candidate.candidate_id}:\nModel benchmark time: {candidate.model_benchmark_time} on device {candidate.model_benchmark_device_id}\nDispatch benchmark time: {candidate.first_benchmark_time} on device {candidate.model_benchmark_device_id}\nSpec file path: {candidate.spec_path}\nSpec contents:{config_file_contents}\n\n"
        dump_list.append(final_str)

    with open(path_config.result_summary_log, "w") as file:
        file.writelines(dump_list)


def sanitize_filename(filename: str) -> str:
    # Replace invalid characters by an underscore
    sanitized = re.sub(r"[^\w\.-]", "_", filename)
    return sanitized
