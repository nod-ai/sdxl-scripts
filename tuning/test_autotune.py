import pytest
import autotune

"""
Usage: python -m pytest test_autotune.py
"""


def test_group_benchmark_results_by_device_id():
    def generate_res(res_arg: str, device_id: int) -> autotune.TaskResult:
        result = autotune.subprocess.CompletedProcess(
            args=[res_arg],
            returncode=0,
        )
        return autotune.TaskResult(result=result, device_id=device_id)

    test_input = [
        generate_res("str1", 3),
        generate_res("str7", 4),
        generate_res("str2", 1),
        generate_res("str5", 3),
        generate_res("str5", 7),
        generate_res("str3", 4),
    ]
    expect_output = [
        [generate_res("str2", 1)],
        [generate_res("str1", 3), generate_res("str5", 3)],
        [generate_res("str7", 4), generate_res("str3", 4)],
        [generate_res("str5", 7)],
    ]

    actual_output = autotune.group_benchmark_results_by_device_id(test_input)

    for a, e in zip(actual_output, expect_output):
        for res1, res2 in zip(a, e):
            assert res1.result.args == res2.result.args
            assert res1.device_id == res2.device_id


def test_sort_candidates_by_first_benchmark_times():
    candidate_trackers = [autotune.CandidateTracker(i) for i in range(5)]
    candidate_trackers[0].first_benchmark_time = 35
    candidate_trackers[1].first_benchmark_time = 2141
    candidate_trackers[2].first_benchmark_time = 231
    candidate_trackers[3].first_benchmark_time = 231.23
    candidate_trackers[4].first_benchmark_time = 58
    test_input = [i for i in range(5)]
    expect_output = [0, 4, 2, 3, 1]
    assert (
        autotune.sort_candidates_by_first_benchmark_times(
            test_input, candidate_trackers
        )
        == expect_output
    )


def test_find_collisions():
    input = [(1, "abc"), (2, "def"), (3, "abc")]
    assert autotune.find_collisions(input) == (True, [("abc", [1, 3]), ("def", [2])])


def test_UnetBenchmarkResult_get_calibrated_result_str():
    baseline_time = 423
    res_time = 304
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(change)
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (-28.132%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 218
    res_time = 218
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(change)
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (+0.000%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 123
    res_time = 345
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(change)
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (+180.488%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

def test_parse_dispatch_benchmark_results():
    def generate_res(stdout: str) -> autotune.TaskResult:
        result = autotune.subprocess.CompletedProcess(
            args=[""],
            stdout=stdout,
            returncode=0,
        )
        return autotune.TaskResult(result)

    def generate_parsed_disptach_benchmark_result(time: float, i: int) -> autotune.parsed_disptach_benchmark_result:
        return autotune.parsed_disptach_benchmark_result(time, path_config.get_candidate_mlir_path(i), path_config.get_candidate_spec_mlir_path(i))

    test_list = [(0, 369.0), (1, 301.0), (2, 457.0), (3, 322.0), (4, 479.0)]
    random_order = [2, 0, 3, 1, 4]
    total = 5
    
    benchmark_results = [generate_res(f"{test_list[i][0]}	Mean Time: {test_list[i][1]}") for i in random_order]

    candidate_trackers = [autotune.CandidateTracker(i) for i in range(total)]
    expect_candidate_trackers = [autotune.CandidateTracker(i) for i in range(total)]

    for i in range(total):
        expect_candidate_trackers[test_list[i][0]].first_benchmark_time = test_list[i][1]

    path_config = autotune.PathConfig()

    tmp = [generate_parsed_disptach_benchmark_result(t, i) for i, t in test_list]
    expect_parsed_results = [tmp[i] for i in random_order]
    expect_dump_list = [f"{test_list[i][0]}	Mean Time: {test_list[i][1]}" for i in random_order]

    parsed_results, dump_list = autotune.parse_dispatch_benchmark_results(path_config, benchmark_results, candidate_trackers)

    assert parsed_results == expect_parsed_results
    assert dump_list == expect_dump_list