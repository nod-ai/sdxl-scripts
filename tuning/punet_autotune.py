import autotune
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PunetClient(autotune.TuningClient):

    def get_dispatch_compile_command(self, candidate_tracker: autotune.CandidateTracker) -> list[str]:
        mlir_path = candidate_tracker.mlir_path
        assert mlir_path is not None
        command = [
            "./compile_candidate.sh",
            "winograd",
            mlir_path.as_posix(),
        ]
        return command

    def get_dispatch_benchmark_command(self, candidate_tracker: autotune.CandidateTracker) -> list[str]:
        pass

    def get_model_compile_command(self, candidate_tracker: autotune.CandidateTracker) -> list[str]:
        pass

    def get_model_benchmark_command(self, candidate_tracker: autotune.CandidateTracker) -> list[str]:
        pass

    def get_compiled_file_index(self, file_name: Path) -> int:
       return int(file_name.stem)


def main():
    args = autotune.parse_arguments()
    path_config = autotune.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[autotune.CandidateTracker] = []
    punet_candidates = PunetClient()

    # path_config.compile_model_sh = Path("./compile_unet_candidate.sh")
    # path_config.benchmark_model_sh = Path("./benchmark_unet_candidate.sh")
    # path_config.model_baseline_vmfb = Path("./unet_baseline.vmfb")

    autotune.setup_logging(args, path_config)

    candidates = autotune.generate_candidates(args, path_config, candidate_trackers)

    compiled_candidates = autotune.compile_dispatches(
        args, path_config, candidates, candidate_trackers, punet_candidates
    )
    
    exit()
    top_candidates = autotune.benchmark_dispatches(
        args, path_config, compiled_candidates, candidate_trackers
    )

    punet_candidates = autotune.compile_models(
        args, path_config, top_candidates, candidate_trackers
    )

    autotune.benchmark_models(args, path_config, punet_candidates, candidate_trackers)

if __name__ == "__main__":
    main()
