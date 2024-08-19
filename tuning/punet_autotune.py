import autotune
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
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

    autotune.setup_logging(args, path_config)

    candidates = autotune.generate_candidates(
        args, path_config, candidate_trackers, punet_client
    )

    compiled_candidates = autotune.compile_dispatches(
        args, path_config, candidates, candidate_trackers, punet_client
    )

    top_candidates = autotune.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers, punet_client
    )

    punet_candidates = autotune.compile_models(
        args, path_config, top_candidates, candidate_trackers, punet_client
    )

    autotune.benchmark_models(
        args, path_config, punet_candidates, candidate_trackers, punet_client
    )


if __name__ == "__main__":
    main()
