import subprocess
from dataclasses import dataclass


@dataclass
class CompileDispatchTaskPack:
    candidate_id: int
    input_path_str: str
    out_path_str: str
    mode: str


@dataclass
class CompileDispatchResultPack:
    candidate_id: int
    success: bool


def compile_dispatch(
    compile_dispatch_task_pack: CompileDispatchTaskPack,
) -> CompileDispatchResultPack:
    candidate_id = compile_dispatch_task_pack.candidate_id
    input_path_str = compile_dispatch_task_pack.input_path_str
    out_path_str = compile_dispatch_task_pack.out_path_str
    mode = compile_dispatch_task_pack.mode

    try:
        subprocess.run(
            [
                "timeout",
                "4s",
                "./punet.sh",
                input_path_str,
                "-o",
                out_path_str,
                "--compile-from=executable-sources",
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return CompileDispatchResultPack(success=False, candidate_id=candidate_id)

    # Check for 'rocm-hsaco-fb' in the output
    try:
        result = subprocess.run(
            ["tools/iree-dump-module", out_path_str],
            capture_output=True,
            text=True,
            check=True,
        )
        if "rocm-hsaco-fb" not in result.stdout:
            raise RuntimeError
    except (subprocess.CalledProcessError, RuntimeError):
        return CompileDispatchResultPack(success=False, candidate_id=candidate_id)

    # TODO: Add local logger to print this message
    # print(f"Compiling {candidate_id}: success")
    return CompileDispatchResultPack(success=True, candidate_id=candidate_id)


def benchmark_dispatch():
    # TODO
    pass


def compile_model():
    # TODO
    pass


def benchmark_model():
    # TODO
    pass


def main():
    return


if __name__ == "__main__":
    main()
