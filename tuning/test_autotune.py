import pytest
import autotune

"""
Usage: python -m pytest test_autotune.py
"""


def test_group_benchmark_results_by_device_id():
    def generate_res(res_arg: str, device_id: int) -> autotune.TaskResult:
        result: autotune.subprocess.CompletedProcess = autotune.subprocess.CompletedProcess(
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
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert autotune.find_collisions(input) == (
        False,
        [("abc", [1]), ("def", [2]), ("hig", [3])],
    )


def test_collision_handler():
    input = [(1, "abc"), (2, "def"), (3, "abc"), (4, "def"), (5, "hig")]
    assert autotune.collision_handler(input) == (True, [1, 2, 5])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert autotune.collision_handler(input) == (False, [])


def test_DispatchBenchmarkResult_get():
    normal_str = "2	Mean Time: 586.0"
    res = autotune.DispatchBenchmarkResult(normal_str)
    assert res.result_str == normal_str
    assert res.get_tokens() == ["2", "Mean", "Time:", "586.0"]
    assert res.get_candidate_id() == 2
    assert res.get_benchmark_time() == 586.0

    incomplete_str = "2	Mean Time:"
    res = autotune.DispatchBenchmarkResult(incomplete_str)
    assert res.get_tokens() == ["2", "Mean", "Time:"]
    assert res.get_candidate_id() == 2
    assert res.get_benchmark_time() == None
    incomplete_str = ""
    res = autotune.DispatchBenchmarkResult(incomplete_str)
    assert res.get_tokens() == []
    assert res.get_candidate_id() == None
    assert res.get_benchmark_time() == None

    bad_str = 12345
    res = autotune.DispatchBenchmarkResult(bad_str)
    assert res.get_tokens() == []
    assert res.get_candidate_id() == None
    assert res.get_benchmark_time() == None


def test_UnetBenchmarkResult_get():
    normal_str = "Benchmarking: unet_candidate_12.vmfb on device 24\nBM_main/process_time/real_time_median 182 ms 183 ms 5 items_per_second=5.50302/s"
    res = autotune.UnetBenchmarkResult(normal_str)
    assert res.result_str == normal_str
    assert res.get_tokens() == [
        "Benchmarking:",
        "unet_candidate_12.vmfb",
        "on",
        "device",
        "24",
        "BM_main/process_time/real_time_median",
        "182",
        "ms",
        "183",
        "ms",
        "5",
        "items_per_second=5.50302/s",
    ]
    assert res.get_unet_candidate_path() == "unet_candidate_12.vmfb"
    assert res.get_candidate_id() == 12
    assert res.get_device_id() == 24
    assert res.get_benchmark_time() == 182.0

    incomplete_str = "Benchmarking: unet_baseline.vmfb on device 24\n"
    res = autotune.UnetBenchmarkResult(incomplete_str)
    assert res.get_tokens() == [
        "Benchmarking:",
        "unet_baseline.vmfb",
        "on",
        "device",
        "24",
    ]
    assert res.get_unet_candidate_path() == "unet_baseline.vmfb"
    assert res.get_candidate_id() == None
    assert res.get_device_id() == 24
    assert res.get_benchmark_time() == None
    incomplete_str = ""
    res = autotune.UnetBenchmarkResult(incomplete_str)
    assert res.get_tokens() == []
    assert res.get_unet_candidate_path() == None
    assert res.get_candidate_id() == None
    assert res.get_device_id() == None
    assert res.get_benchmark_time() == None

    bad_str = 12345
    res = autotune.UnetBenchmarkResult(bad_str)
    assert res.get_tokens() == []
    assert res.get_unet_candidate_path() == None
    assert res.get_candidate_id() == None
    assert res.get_device_id() == None
    assert res.get_benchmark_time() == None


def test_generate_sample_result():
    res = autotune.DispatchBenchmarkResult()
    output = res.generate_sample_result(1, 3.14)
    expected = f"1\tMean Time: 3.1\n"
    assert output == expected, "DispatchBenchmarkResult generates invalid sample string"

    res = autotune.UnetBenchmarkResult()
    output = res.generate_sample_result(
        1, "some_dir/tuning_2024_07_24_20_06/unet_candidate_60.vmfb.vmfb", 576.89
    )
    expected = f"Benchmarking: 1 on device some_dir/tuning_2024_07_24_20_06/unet_candidate_60.vmfb.vmfb\nBM_run_forward/process_time/real_time_median\t    577 ms\t    578 ms\t      5 items_per_second=2.884450/s\n\n"
    assert output == expected, "UnetBenchmarkResult generates invalid sample string"


def test_UnetBenchmarkResult_get_calibrated_result_str():
    baseline_time = 423
    res_time = 304
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (-28.132%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 218
    res_time = 218
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (+0.000%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 123
    res_time = 345
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = autotune.UnetBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
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

    def generate_parsed_disptach_benchmark_result(
        time: float, i: int
    ) -> autotune.ParsedDisptachBenchmarkResult:
        return autotune.ParsedDisptachBenchmarkResult(
            time,
            path_config.get_candidate_mlir_path(i),
            path_config.get_candidate_spec_mlir_path(i),
        )

    test_list = [(0, 369.0), (1, 301.0), (2, 457.0), (3, 322.0), (4, 479.0)]
    random_order = [2, 0, 3, 1, 4]
    total = 5

    benchmark_results = [
        generate_res(f"{test_list[i][0]}	Mean Time: {test_list[i][1]}")
        for i in random_order
    ]

    candidate_trackers = [autotune.CandidateTracker(i) for i in range(total)]
    candidate_trackers_before = [autotune.CandidateTracker(i) for i in range(total)]

    expect_candidate_trackers = [autotune.CandidateTracker(i) for i in range(total)]
    for i in range(total):
        expect_candidate_trackers[test_list[i][0]].first_benchmark_time = test_list[i][
            1
        ]

    path_config = autotune.PathConfig()

    tmp = [generate_parsed_disptach_benchmark_result(t, i) for i, t in test_list]
    expect_parsed_results = [tmp[i] for i in random_order]
    expect_dump_list = [
        f"{test_list[i][0]}	Mean Time: {test_list[i][1]}" for i in random_order
    ]

    parsed_results, dump_list = autotune.parse_dispatch_benchmark_results(
        path_config, benchmark_results, candidate_trackers
    )

    assert parsed_results == expect_parsed_results
    assert dump_list == expect_dump_list
    assert candidate_trackers != candidate_trackers_before
    assert candidate_trackers == expect_candidate_trackers


def test_parse_grouped_benchmark_results():
    def generate_res(stdout: str, device_id: int) -> autotune.TaskResult:
        result = autotune.subprocess.CompletedProcess(
            args=[""],
            stdout=stdout,
            returncode=0,
        )
        return autotune.TaskResult(result=result, device_id=device_id)

    def set_tracker(
        tracker: autotune.CandidateTracker,
        unet_benchmark_time: float,
        unet_benchmark_device_id: int,
        baseline_benchmark_time: float,
        calibrated_benchmark_diff=float,
    ):
        tracker.unet_benchmark_time = unet_benchmark_time
        tracker.unet_benchmark_device_id = unet_benchmark_device_id
        tracker.baseline_benchmark_time = baseline_benchmark_time
        tracker.calibrated_benchmark_diff = calibrated_benchmark_diff

    b1 = "Benchmarking: some_dir/unet_baseline.vmfb on device 0 BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s"
    b2 = "Benchmarking: unet_baseline.vmfb on device 1 BM_main/process_time/real_time_median 59.8 ms 15.1 ms 5 items_per_second=16.7114/s"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0 BM_main/process_time/real_time_median 62.4 ms 15.4 ms 5 items_per_second=16.0223/s"
    s2 = "Benchmarking: some_dir/unet_candidate_2.vmfb on device 1 BM_main/process_time/real_time_median 61.4 ms 11.0 ms 5 items_per_second=16.2958/s"
    s3 = "Benchmarking: unet_candidate_4.vmfb on device 1 BM_main/process_time/real_time_median 57.4 ms 11.0 ms 5 items_per_second=16.2958/s"

    grouped_benchmark_results = [
        [generate_res(b1, 0), generate_res(s1, 0)],
        [
            generate_res(b2, 1),
            generate_res(None, 1),
            generate_res(s2, 1),
            generate_res(s3, 1),
        ],
    ]

    path_config = autotune.PathConfig()

    candidate_trackers = [autotune.CandidateTracker(i) for i in range(5)]

    candidate_trackers_before = [autotune.CandidateTracker(i) for i in range(5)]
    expect_candidate_trackers = [autotune.CandidateTracker(i) for i in range(5)]
    set_tracker(expect_candidate_trackers[1], 62.4, 0, 60.7, 0.028006589785831888)
    set_tracker(expect_candidate_trackers[2], 61.4, 1, 59.8, 0.02675585284280939)
    set_tracker(expect_candidate_trackers[4], 57.4, 1, 59.8, -0.04013377926421403)

    expect_dump_list = [
        "Benchmarking: some_dir/unet_baseline.vmfb on device 0 "
        "BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s",
        "Benchmarking: unet_candidate_1.vmfb on device 0 "
        "BM_main/process_time/real_time_median 62.4 ms (+2.801%) 15.4 ms 5 items_per_second=16.0223/s",
        "Benchmarking: unet_baseline.vmfb on device 1 "
        "BM_main/process_time/real_time_median 59.8 ms 15.1 ms 5 items_per_second=16.7114/s",
        "Benchmarking: unet_candidate_4.vmfb on device 1 "
        "BM_main/process_time/real_time_median 57.4 ms (-4.013%) 11.0 ms 5 items_per_second=16.2958/s",
        "Benchmarking: some_dir/unet_candidate_2.vmfb on device 1 "
        "BM_main/process_time/real_time_median 61.4 ms (+2.676%) 11.0 ms 5 items_per_second=16.2958/s",
    ]

    dump_list = autotune.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )

    assert dump_list == expect_dump_list, "basic parsing is incorrect"
    assert (
        candidate_trackers != candidate_trackers_before
    ), "candidate_trackers should be modified"
    assert (
        candidate_trackers == expect_candidate_trackers
    ), "candidate_trackers did not change as expected"

    b1 = "Benchmarking: unet_baseline.vmfb on device 0"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0 BM_main/process_time/real_time_median 62.4 ms 15.4 ms 5 items_per_second=16.0223/s"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    dump_list = autotune.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking: unet_candidate_1.vmfb on device 0 "
        "BM_main/process_time/real_time_median 62.4 ms 15.4 ms 5 items_per_second=16.0223/s",
        "Benchmarking result of unet_baseline.vmfb on deivce 0 is incomplete\n",
    ]
    assert dump_list == expect_dump_list, "fail to parse incomplete baselines"

    b1 = "Benchmarking: some_dir/unet_baseline.vmfb on device 0 BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    candidate_trackers[1].unet_candidate_path = "unet_candidate_1.vmfb"
    dump_list = autotune.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking: some_dir/unet_baseline.vmfb on device 0 "
        "BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s",
        "Benchmarking result of unet_candidate_1.vmfb on deivce 0 is incomplete\n",
    ]
    assert dump_list == expect_dump_list, "fail to parse incomplete candidates"

    b1 = "Benchmarking: unet_baseline.vmfb on device 0"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    candidate_trackers[1].unet_candidate_path = "unet_candidate_1.vmfb"
    dump_list = autotune.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking result of unet_baseline.vmfb on deivce 0 is incomplete\n",
        "Benchmarking result of unet_candidate_1.vmfb on deivce 0 is incomplete\n",
    ]
    assert (
        dump_list == expect_dump_list
    ), "fail to parse incomplete baseline and candidates"
