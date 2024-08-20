# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pytest
from unittest.mock import call, patch, MagicMock
import libtuner

"""
Usage: python -m pytest test_libtuner.py
"""


def test_group_benchmark_results_by_device_id():
    # Create mock TaskResult objects with device_id attributes
    task_result_1 = MagicMock()
    task_result_1.device_id = "device_1"

    task_result_2 = MagicMock()
    task_result_2.device_id = "device_2"

    task_result_3 = MagicMock()
    task_result_3.device_id = "device_1"

    benchmark_results = [task_result_1, task_result_2, task_result_3]

    expected_grouped_results = [
        [task_result_1, task_result_3],  # Grouped by device_1
        [task_result_2],  # Grouped by device_2
    ]

    grouped_results = libtuner.group_benchmark_results_by_device_id(benchmark_results)

    assert grouped_results == expected_grouped_results
    assert grouped_results[0][0].device_id == "device_1"
    assert grouped_results[1][0].device_id == "device_2"


def test_sort_candidates_by_first_benchmark_times():
    candidate_trackers = [libtuner.CandidateTracker(i) for i in range(5)]
    candidate_trackers[0].first_benchmark_time = 35
    candidate_trackers[1].first_benchmark_time = 2141
    candidate_trackers[2].first_benchmark_time = 231
    candidate_trackers[3].first_benchmark_time = 231.23
    candidate_trackers[4].first_benchmark_time = 58
    test_input = [i for i in range(5)]
    expect_output = [0, 4, 2, 3, 1]
    assert (
        libtuner.sort_candidates_by_first_benchmark_times(
            test_input, candidate_trackers
        )
        == expect_output
    )


def test_find_collisions():
    input = [(1, "abc"), (2, "def"), (3, "abc")]
    assert libtuner.find_collisions(input) == (True, [("abc", [1, 3]), ("def", [2])])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.find_collisions(input) == (
        False,
        [("abc", [1]), ("def", [2]), ("hig", [3])],
    )


def test_collision_handler():
    input = [(1, "abc"), (2, "def"), (3, "abc"), (4, "def"), (5, "hig")]
    assert libtuner.collision_handler(input) == (True, [1, 2, 5])
    input = [(1, "abc"), (2, "def"), (3, "hig")]
    assert libtuner.collision_handler(input) == (False, [])


def test_DispatchBenchmarkResult_get():
    # Time is int
    normal_str = r"""
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                                                      Time             CPU   Iterations UserCounters...
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               274 us          275 us         3000 items_per_second=3.65611k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               274 us          275 us         3000 items_per_second=3.65481k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               273 us          275 us         3000 items_per_second=3.65671k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_mean          274 us          275 us            3 items_per_second=3.65587k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_median        274 us          275 us            3 items_per_second=3.65611k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_stddev      0.073 us        0.179 us            3 items_per_second=0.971769/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_cv           0.03 %          0.07 %             3 items_per_second=0.03%
    """
    res = libtuner.DispatchBenchmarkResult(candidate_id=1, result_str=normal_str)
    assert res.get_benchmark_time() == float(274)

    # Time is float
    res = libtuner.DispatchBenchmarkResult(
        candidate_id=2, result_str="process_time/real_time_mean 123.45 us"
    )
    assert res.get_benchmark_time() == 123.45

    # Invalid str
    res = libtuner.DispatchBenchmarkResult(candidate_id=3, result_str="hello world")
    assert res.get_benchmark_time() == None
    res = libtuner.DispatchBenchmarkResult(candidate_id=4, result_str="")
    assert res.get_benchmark_time() == None


def test_ModelBenchmarkResult_get():
    normal_str = "Benchmarking: unet_candidate_12.vmfb on device 24\nBM_main/process_time/real_time_median 182 ms 183 ms 5 items_per_second=5.50302/s"
    res = libtuner.ModelBenchmarkResult(normal_str)
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
    assert res.get_model_candidate_path() == "unet_candidate_12.vmfb"
    assert res.get_candidate_id() == 12
    assert res.get_device_id() == 24
    assert res.get_benchmark_time() == 182.0

    incomplete_str = "Benchmarking: baseline.vmfb on device 24\n"
    res = libtuner.ModelBenchmarkResult(incomplete_str)
    assert res.get_tokens() == [
        "Benchmarking:",
        "baseline.vmfb",
        "on",
        "device",
        "24",
    ]
    assert res.get_model_candidate_path() == "baseline.vmfb"
    assert res.get_candidate_id() == None
    assert res.get_device_id() == 24
    assert res.get_benchmark_time() == None
    incomplete_str = ""
    res = libtuner.ModelBenchmarkResult(incomplete_str)
    assert res.get_tokens() == []
    assert res.get_model_candidate_path() == None
    assert res.get_candidate_id() == None
    assert res.get_device_id() == None
    assert res.get_benchmark_time() == None

    bad_str = 12345
    res = libtuner.ModelBenchmarkResult(bad_str)
    assert res.get_tokens() == []
    assert res.get_model_candidate_path() == None
    assert res.get_candidate_id() == None
    assert res.get_device_id() == None
    assert res.get_benchmark_time() == None


def test_generate_sample_result():
    output = libtuner.generate_sample_DBR(1, 3.14)
    expected = f"1\tMean Time: 3.1\n"
    assert output == expected, "DispatchBenchmarkResult generates invalid sample string"

    res = libtuner.ModelBenchmarkResult()
    output = res.generate_sample_result(
        1, "some_dir/tuning_2024_07_24_20_06/unet_candidate_60.vmfb.vmfb", 576.89
    )
    expected = f"Benchmarking: 1 on device some_dir/tuning_2024_07_24_20_06/unet_candidate_60.vmfb.vmfb\nBM_run_forward/process_time/real_time_median\t    577 ms\t    578 ms\t      5 items_per_second=2.884450/s\n\n"
    assert output == expected, "UnetBenchmarkResult generates invalid sample string"


def test_ModelBenchmarkResult_get_calibrated_result_str():
    baseline_time = 423
    res_time = 304
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = libtuner.ModelBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (-28.132%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 218
    res_time = 218
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = libtuner.ModelBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (+0.000%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str

    baseline_time = 123
    res_time = 345
    result_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median	    {float(res_time)} ms	    305 ms	      5 items_per_second=1.520000/s"
    change = (res_time - baseline_time) / baseline_time
    output_str = libtuner.ModelBenchmarkResult(result_str).get_calibrated_result_str(
        change
    )
    expect_str = f"Benchmarking: tuning_2024_07_22_16_29/unet_candidate_16.vmfb on device 0\nBM_run_forward/process_time/real_time_median\t    {float(res_time)} ms (+180.488%)\t    305 ms\t      5 items_per_second=1.520000/s"
    assert output_str == expect_str


def test_parse_dispatch_benchmark_results():
    base_path = libtuner.Path("/mock/base/dir")
    spec_dir = base_path / "specs"
    path_config = libtuner.PathConfig()
    object.__setattr__(path_config, "spec_dir", spec_dir)

    mock_result_1 = MagicMock()
    mock_result_1.result.stdout = "process_time/real_time_mean 100.0 us"
    mock_result_1.candidate_id = 1
    mock_result_2 = MagicMock()
    mock_result_2.result.stdout = "process_time/real_time_mean 200.0 us"
    mock_result_2.candidate_id = 2
    benchmark_results = [mock_result_1, mock_result_2]

    candidate_tracker_0 = libtuner.CandidateTracker(candidate_id=0)
    candidate_tracker_0.dispatch_mlir_path = libtuner.Path("/mock/mlir/path/0.mlir")
    candidate_tracker_1 = libtuner.CandidateTracker(candidate_id=1)
    candidate_tracker_1.dispatch_mlir_path = libtuner.Path("/mock/mlir/path/1.mlir")
    candidate_tracker_2 = libtuner.CandidateTracker(candidate_id=2)
    candidate_tracker_2.dispatch_mlir_path = libtuner.Path("/mock/mlir/path/2.mlir")
    candidate_trackers = [candidate_tracker_0, candidate_tracker_1, candidate_tracker_2]

    expected_parsed_results = [
        libtuner.ParsedDisptachBenchmarkResult(
            candidate_id=1,
            benchmark_time_in_seconds=100.0,
            candidate_mlir=libtuner.Path("/mock/mlir/path/1.mlir"),
            candidate_spec_mlir=libtuner.Path("/mock/base/dir/specs/1_spec.mlir"),
        ),
        libtuner.ParsedDisptachBenchmarkResult(
            candidate_id=2,
            benchmark_time_in_seconds=200.0,
            candidate_mlir=libtuner.Path("/mock/mlir/path/2.mlir"),
            candidate_spec_mlir=libtuner.Path("/mock/base/dir/specs/2_spec.mlir"),
        ),
    ]
    expected_dump_list = [
        "process_time/real_time_mean 100.0 us",
        "process_time/real_time_mean 200.0 us",
    ]

    parsed_results, dump_list = libtuner.parse_dispatch_benchmark_results(
        path_config, benchmark_results, candidate_trackers
    )

    assert parsed_results == expected_parsed_results
    assert dump_list == expected_dump_list
    assert candidate_trackers[1].first_benchmark_time == 100.0
    assert candidate_trackers[1].spec_path == libtuner.Path(
        "/mock/base/dir/specs/1_spec.mlir"
    )
    assert candidate_trackers[2].first_benchmark_time == 200.0
    assert candidate_trackers[2].spec_path == libtuner.Path(
        "/mock/base/dir/specs/2_spec.mlir"
    )


def test_parse_grouped_benchmark_results():
    def generate_res(stdout: str, device_id: int) -> libtuner.TaskResult:
        result = libtuner.subprocess.CompletedProcess(
            args=[""],
            stdout=stdout,
            returncode=0,
        )
        candidate_id = 0
        return libtuner.TaskResult(
            result=result, candidate_id=candidate_id, device_id=str(device_id)
        )

    def set_tracker(
        tracker: libtuner.CandidateTracker,
        model_benchmark_time: float,
        model_benchmark_device_id: int,
        baseline_benchmark_time: float,
        calibrated_benchmark_diff=float,
    ):
        tracker.model_benchmark_time = model_benchmark_time
        tracker.model_benchmark_device_id = model_benchmark_device_id
        tracker.baseline_benchmark_time = baseline_benchmark_time
        tracker.calibrated_benchmark_diff = calibrated_benchmark_diff

    b1 = "Benchmarking: some_dir/baseline.vmfb on device 0 BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s"
    b2 = "Benchmarking: baseline.vmfb on device 1 BM_main/process_time/real_time_median 59.8 ms 15.1 ms 5 items_per_second=16.7114/s"
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

    path_config = libtuner.PathConfig()

    candidate_trackers = [libtuner.CandidateTracker(i) for i in range(5)]

    candidate_trackers_before = [libtuner.CandidateTracker(i) for i in range(5)]
    expect_candidate_trackers = [libtuner.CandidateTracker(i) for i in range(5)]
    set_tracker(expect_candidate_trackers[1], 62.4, 0, 60.7, 0.028006589785831888)
    set_tracker(expect_candidate_trackers[2], 61.4, 1, 59.8, 0.02675585284280939)
    set_tracker(expect_candidate_trackers[4], 57.4, 1, 59.8, -0.04013377926421403)

    expect_dump_list = [
        "Benchmarking: some_dir/baseline.vmfb on device 0 "
        "BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s",
        "Benchmarking: unet_candidate_1.vmfb on device 0 "
        "BM_main/process_time/real_time_median 62.4 ms (+2.801%) 15.4 ms 5 items_per_second=16.0223/s",
        "Benchmarking: baseline.vmfb on device 1 "
        "BM_main/process_time/real_time_median 59.8 ms 15.1 ms 5 items_per_second=16.7114/s",
        "Benchmarking: unet_candidate_4.vmfb on device 1 "
        "BM_main/process_time/real_time_median 57.4 ms (-4.013%) 11.0 ms 5 items_per_second=16.2958/s",
        "Benchmarking: some_dir/unet_candidate_2.vmfb on device 1 "
        "BM_main/process_time/real_time_median 61.4 ms (+2.676%) 11.0 ms 5 items_per_second=16.2958/s",
    ]

    dump_list = libtuner.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )

    assert dump_list == expect_dump_list, "basic parsing is incorrect"
    assert (
        candidate_trackers != candidate_trackers_before
    ), "candidate_trackers should be modified"
    assert (
        candidate_trackers == expect_candidate_trackers
    ), "candidate_trackers did not change as expected"

    b1 = "Benchmarking: baseline.vmfb on device 0"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0 BM_main/process_time/real_time_median 62.4 ms 15.4 ms 5 items_per_second=16.0223/s"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    dump_list = libtuner.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking: unet_candidate_1.vmfb on device 0 "
        "BM_main/process_time/real_time_median 62.4 ms 15.4 ms 5 items_per_second=16.0223/s",
        "Benchmarking result of baseline.vmfb on deivce 0 is incomplete\n",
    ]
    assert dump_list == expect_dump_list, "fail to parse incomplete baselines"

    b1 = "Benchmarking: some_dir/baseline.vmfb on device 0 BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    candidate_trackers[1].model_path = "unet_candidate_1.vmfb"
    dump_list = libtuner.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking: some_dir/baseline.vmfb on device 0 "
        "BM_main/process_time/real_time_median 60.7 ms 13.5 ms 5 items_per_second=16.4733/s",
        "Benchmarking result of unet_candidate_1.vmfb on deivce 0 is incomplete\n",
    ]
    assert dump_list == expect_dump_list, "fail to parse incomplete candidates"

    b1 = "Benchmarking: baseline.vmfb on device 0"
    s1 = "Benchmarking: unet_candidate_1.vmfb on device 0"
    grouped_benchmark_results = [[generate_res(b1, 0), generate_res(s1, 0)]]
    candidate_trackers[1].model_path = "unet_candidate_1.vmfb"
    dump_list = libtuner.parse_grouped_benchmark_results(
        path_config, grouped_benchmark_results, candidate_trackers
    )
    expect_dump_list = [
        "Benchmarking result of baseline.vmfb on deivce 0 is incomplete\n",
        "Benchmarking result of unet_candidate_1.vmfb on deivce 0 is incomplete\n",
    ]
    assert (
        dump_list == expect_dump_list
    ), "fail to parse incomplete baseline and candidates"


def test_extract_driver_names():
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    expected_output = {"hip", "local-sync", "cuda"}

    assert libtuner.extract_driver_names(user_devices) == expected_output


def test_fetch_available_devices_success():
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "0"}],
        "local-sync": [{"path": "default"}],
        "cuda": [{"path": "default"}],
    }

    with patch("libtuner.ireert.get_driver") as mock_get_driver:
        mock_driver = MagicMock()

        def get_mock_driver(name):
            mock_driver.query_available_devices.side_effect = lambda: mock_devices[name]
            return mock_driver

        mock_get_driver.side_effect = get_mock_driver

        actual_output = libtuner.fetch_available_devices(drivers)
        expected_output = ["hip://0", "local-sync://default", "cuda://default"]

        assert actual_output == expected_output


def test_fetch_available_devices_failure():
    drivers = ["hip", "local-sync", "cuda"]
    mock_devices = {
        "hip": [{"path": "0"}],
        "local-sync": ValueError("Failed to initialize"),
        "cuda": [{"path": "default"}],
    }

    with patch("libtuner.ireert.get_driver") as mock_get_driver:
        with patch("libtuner.handle_error") as mock_handle_error:
            mock_driver = MagicMock()

            def get_mock_driver(name):
                if isinstance(mock_devices[name], list):
                    mock_driver.query_available_devices.side_effect = (
                        lambda: mock_devices[name]
                    )
                else:
                    mock_driver.query_available_devices.side_effect = lambda: (
                        _ for _ in ()
                    ).throw(mock_devices[name])
                return mock_driver

            mock_get_driver.side_effect = get_mock_driver

            actual_output = libtuner.fetch_available_devices(drivers)
            expected_output = ["hip://0", "cuda://default"]

            assert actual_output == expected_output
            mock_handle_error.assert_called_once_with(
                condition=True,
                msg="Could not initialize driver local-sync: Failed to initialize",
                error_type=ValueError,
                exit_program=True,
            )


def test_parse_devices():
    user_devices_str = "hip://0, local-sync://default, cuda://default"
    expected_output = ["hip://0", "local-sync://default", "cuda://default"]

    with patch("libtuner.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_not_called()


def test_parse_devices_with_invalid_input():
    user_devices_str = "hip://0, local-sync://default, invalid_device, cuda://default"
    expected_output = [
        "hip://0",
        "local-sync://default",
        "invalid_device",
        "cuda://default",
    ]

    with patch("libtuner.handle_error") as mock_handle_error:
        actual_output = libtuner.parse_devices(user_devices_str)
        assert actual_output == expected_output

        mock_handle_error.assert_called_once_with(
            condition=True,
            msg=f"Invalid device list: {user_devices_str}. Error: {ValueError()}",
            error_type=argparse.ArgumentTypeError,
        )


def test_validate_devices():
    user_devices = ["hip://0", "local-sync://default"]
    user_drivers = {"hip", "local-sync"}

    with patch("libtuner.extract_driver_names", return_value=user_drivers):
        with patch(
            "libtuner.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch("libtuner.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                assert all(
                    call[1]["condition"] is False
                    for call in mock_handle_error.call_args_list
                )


def test_validate_devices_with_invalid_device():
    user_devices = ["hip://0", "local-sync://default", "cuda://default"]
    user_drivers = {"hip", "local-sync", "cuda"}

    with patch("libtuner.extract_driver_names", return_value=user_drivers):
        with patch(
            "libtuner.fetch_available_devices",
            return_value=["hip://0", "local-sync://default"],
        ):
            with patch("libtuner.handle_error") as mock_handle_error:
                libtuner.validate_devices(user_devices)
                expected_call = call(
                    condition=True,
                    msg=f"Invalid device specified: cuda://default\nFetched available devices: ['hip://0', 'local-sync://default']",
                    error_type=argparse.ArgumentError,
                    exit_program=True,
                )
                assert expected_call in mock_handle_error.call_args_list
