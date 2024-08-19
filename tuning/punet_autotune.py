# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import autotune
from pathlib import Path


class PunetClient(autotune.TuningClient):

    def get_dispatch_compile_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        mlir_path = candidate_tracker.dispatch_mlir_path
        assert mlir_path is not None
        command = [
            "./compile_candidate.sh",
            "winograd",
            mlir_path.as_posix(),
        ]
        return command

    def get_dispatch_benchmark_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        compiled_vmfb_path = candidate_tracker.compiled_dispatch_path
        assert compiled_vmfb_path is not None
        command = [
            "./benchmark_dispatch.sh",
            compiled_vmfb_path.as_posix(),
        ]
        return command

    def get_model_compile_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        mlir_spec_path = candidate_tracker.spec_path
        assert mlir_spec_path is not None
        command = [
            "./compile_unet_candidate.sh",
            "winograd",
            mlir_spec_path.as_posix(),
        ]
        return command

    def get_model_benchmark_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        unet_candidate_path = candidate_tracker.model_path
        assert unet_candidate_path is not None
        command = [
            "./benchmark_unet_candidate.sh",
            unet_candidate_path.as_posix(),
        ]
        return command


def main():
    args = autotune.parse_arguments()
    path_config = autotune.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[autotune.CandidateTracker] = []
    punet_client = PunetClient()
    stop_after_phase: str = args.stop_after

    print("Setup logging")
    autotune.setup_logging(args, path_config)
    print(path_config.run_log, end="\n\n")

    print("Validating devices")
    autotune.validate_devices(args.devices)
    print("Validation successful!\n")

    print("Generating candidates...")
    candidates = autotune.generate_candidates(
        args, path_config, candidate_trackers, punet_client
    )
    print(f"Generated [{len(candidates)}] candidates in {path_config.candidates_dir}\n")
    if stop_after_phase == autotune.ExecutionPhases.generate_candidates:
        return

    print("Compiling candidates...")
    compiled_candidates = autotune.compile_dispatches(
        args, path_config, candidates, candidate_trackers, punet_client
    )
    print(f"Compiled files are stored in {path_config.compiled_dir}\n")
    if stop_after_phase == autotune.ExecutionPhases.compile_dispatches:
        return

    print("Benchmarking compiled candidates...")
    top_candidates = autotune.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers, punet_client
    )
    print(f"Stored results in {path_config.output_unilog}\n")
    if stop_after_phase == autotune.ExecutionPhases.benchmark_dispatches:
        return

    print(f"Compiling top model candidates...")
    punet_candidates = autotune.compile_models(
        args, path_config, top_candidates, candidate_trackers, punet_client
    )
    print(f"Model candidates compiled in {path_config.base_dir}\n")
    if stop_after_phase == autotune.ExecutionPhases.compile_models:
        return

    print("Benchmarking model candidates...")
    autotune.benchmark_models(
        args, path_config, punet_candidates, candidate_trackers, punet_client
    )
    print(f"Stored results in {path_config.output_unilog}")
    if stop_after_phase == autotune.ExecutionPhases.benchmark_models:
        return

    autotune.summerize_top_candidates(path_config, candidate_trackers)
    print(f"Stored top candidates info in {path_config.result_summary_log}\n")

    autotune.save_pickle(path_config.candidate_trackers_pkl, candidate_trackers)
    print(f"Candidate trackers are saved in {path_config.candidate_trackers_pkl}\n")

    print("Check the detailed execution logs in:")
    print(path_config.run_log)

    for candidate in candidate_trackers:
        autotune.logging.debug(candidate)
        if args.verbose:
            print(candidate)


if __name__ == "__main__":
    main()
