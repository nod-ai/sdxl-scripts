import subprocess
import os


def run_tests(test_file):
    print(f"Running tests in {test_file}...")
    result = subprocess.run(
        ["pytest", test_file], capture_output=True, text=True, cwd="tuning"
    )
    print(result.stdout)
    print(result.stderr)
    result.check_returncode()


def main():
    # Get the list of changed files
    changed_files = subprocess.run(
        ["git", "diff", "--name-only", "--cached"], capture_output=True, text=True
    ).stdout.splitlines()

    # Check if any file in the 'tuning' directory has changed
    tuning_files = [f for f in changed_files if f.startswith("tuning/")]

    if any(f.endswith("tune.py") for f in tuning_files):
        run_tests("test_tune.py")
    if any(f.endswith("autotune.py") for f in tuning_files):
        run_tests("test_autotune.py")


if __name__ == "__main__":
    main()
