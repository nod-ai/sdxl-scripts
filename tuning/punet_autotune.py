import autotune
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PunetClient(autotune.TuningClient):

    def get_dispatch_compile_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        mlir_path = candidate_tracker.mlir_path
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
        compiled_vmfb_path = candidate_tracker.compiled_vmfb_path
        assert compiled_vmfb_path is not None
        command = [
            "./benchmark_dispatch.sh",
            compiled_vmfb_path.as_posix(),
        ]
        return command

    def get_model_compile_command(
        self, candidate_tracker: autotune.CandidateTracker
    ) -> list[str]:
        mlir_spec_path = candidate_tracker.mlir_spec_path
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
        unet_candidate_path = candidate_tracker.unet_candidate_path
        assert unet_candidate_path is not None
        command = [
            "./benchmark_unet_candidate.sh",
            unet_candidate_path.as_posix(),
        ]
        return command

    def get_compiled_dispatch_index(self, file_path: Path) -> int:
        return int(file_path.stem)

    def get_candidate_spec_filename(self, candidate_id: int) -> str:
        return f"{candidate_id}_spec.mlir"

    def get_compiled_model_index(self, file_path: Path) -> int:
        return int(file_path.stem.split("_")[-1])


def set_path_config(path_config: autotune.PathConfig) -> None:
    path_config.model_baseline_vmfb = Path("./unet_baseline.vmfb")
    path_config.candidates_dir = path_config.base_dir / "candidates"
    path_config.candidate_configs_pkl = path_config.candidates_dir / "configs.pkl"
    path_config.compiled_dir = path_config.candidates_dir / "compiled"
    path_config.compile_failed_dir = path_config.candidates_dir / "failed"
    path_config.spec_dir = path_config.candidates_dir / "configs"


def main():
    args = autotune.parse_arguments()
    path_config = autotune.PathConfig()
    path_config.base_dir.mkdir(parents=True, exist_ok=True)
    path_config.output_unilog.touch()
    candidate_trackers: list[autotune.CandidateTracker] = []
    punet_client = PunetClient()

    set_path_config(path_config)

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
