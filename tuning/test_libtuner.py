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


def test_IREEBenchmarkResult_get():
    # Time is int
    normal_str = r"""
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Benchmark                                                                                                                                      Time             CPU   Iterations UserCounters...
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               271 us          275 us         3000 items_per_second=3.65611k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               274 us          275 us         3000 items_per_second=3.65481k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time               273 us          275 us         3000 items_per_second=3.65671k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_mean          274 us          275 us            3 items_per_second=3.65587k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_mean        275 us          275 us            3 items_per_second=3.65611k/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_stddev      0.073 us        0.179 us            3 items_per_second=0.971769/s
    BM_main$async_dispatch_311_rocm_hsaco_fb_main$async_dispatch_311_matmul_like_2x1024x1280x5120_i8xi8xi32/process_time/real_time_cv           0.03 %          0.07 %             3 items_per_second=0.03%
    """
    res = libtuner.IREEBenchmarkResult(candidate_id=1, result_str=normal_str)
    assert res.get_mean_time() == float(274)

    # Time is float
    res = libtuner.IREEBenchmarkResult(
        candidate_id=2,
        result_str="process_time/real_time_mean 123.45 us, process_time/real_time_mean 246.78 us",
    )
    assert res.get_mean_time() == 123.45

    # Invalid str
    res = libtuner.IREEBenchmarkResult(candidate_id=3, result_str="hello world")
    assert res.get_mean_time() == None
    res = libtuner.IREEBenchmarkResult(candidate_id=4, result_str="")
    assert res.get_mean_time() == None


def test_generate_display_BR():
    output = libtuner.generate_display_DBR(1, 3.14)
    expected = f"1\tMean Time: 3.1\n"
    assert output == expected, "DispatchBenchmarkResult generates invalid sample string"

    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89)
    expected = "Benchmarking: baseline.vmfb on device 1: 568\n\n"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"
    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89, 0.0314)
    expected = "Benchmarking: baseline.vmfb on device 1: 568 (+3.140%)\n\n"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"
    output = libtuner.generate_display_MBR("baseline.vmfb", str(1), 567.89, -3.14)
    expected = "Benchmarking: baseline.vmfb on device 1: 568 (-314.000%)\n\n"
    assert output == expected, "ModelBenchmarkResult generates invalid sample string"


def test_parse_dispatch_benchmark_results():
    base_path = libtuner.Path("/mock/base/dir")
    spec_dir = base_path / "specs"
    path_config = libtuner.PathConfig()
    object.__setattr__(path_config, "specs_dir", spec_dir)

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
        "1\tMean Time: 100.0\n",
        "2\tMean Time: 200.0\n",
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


def test_parse_model_benchmark_results():
    # Setup mock data for candidate_trackers
    tracker0 = MagicMock(spec=libtuner.CandidateTracker)
    tracker0.compiled_model_path = MagicMock(
        return_value=libtuner.Path("/path/to/baseline.vmfb")
    )

    tracker1 = MagicMock(spec=libtuner.CandidateTracker)
    tracker1.compiled_model_path = MagicMock(
        return_value=libtuner.Path("/path/to/model_1.vmfb")
    )

    tracker2 = MagicMock(spec=libtuner.CandidateTracker)
    tracker2.compiled_model_path = MagicMock(
        return_value=libtuner.Path("/path/to/model_2.vmfb")
    )

    candidate_trackers = [tracker0, tracker1, tracker2]

    # Setup mock data for task results
    result1 = MagicMock(spec=libtuner.TaskResult)
    result1.result = MagicMock(stdout="1.23")
    result1.candidate_id = 1
    result1.device_id = "device1"

    result2 = MagicMock(spec=libtuner.TaskResult)
    result2.result = MagicMock(stdout="4.56")
    result2.candidate_id = 2
    result2.device_id = "device2"

    result3 = MagicMock(spec=libtuner.TaskResult)
    result3.result = MagicMock(stdout="0.98")
    result3.candidate_id = 0
    result3.device_id = "device1"

    result4 = MagicMock(spec=libtuner.TaskResult)
    result4.result = MagicMock(stdout="4.13")
    result4.candidate_id = 0
    result4.device_id = "device2"

    candidate_results = [result1, result2]
    baseline_results = [result3, result4]

    # Mock IREEBenchmarkResult to return float value from stdout
    def mock_get_mean_time(self):
        return float(self.result_str)

    # Mock IREEBenchmarkResult to return specific benchmark times
    with patch("libtuner.IREEBenchmarkResult.get_mean_time", new=mock_get_mean_time):
        # Mock generate_display_MBR to return a fixed display string
        with patch(
            "libtuner.generate_display_MBR",
            side_effect=lambda *args, **kwargs: "display_str",
        ):
            # Mock handle_error to avoid actual logging during tests
            with patch("libtuner.handle_error") as mock_handle_error:
                # Call the function
                result = libtuner.parse_model_benchmark_results(
                    candidate_trackers, candidate_results, baseline_results
                )

                # Verify interactions with candidate_trackers
                assert tracker1.model_benchmark_time == 1.23
                assert tracker1.model_benchmark_device_id == "device1"
                assert tracker1.baseline_benchmark_time == 0.98
                assert tracker1.calibrated_benchmark_diff == pytest.approx(
                    (1.23 - 0.98) / 0.98, rel=1e-6
                )

                assert tracker2.model_benchmark_time == 4.56
                assert tracker2.model_benchmark_device_id == "device2"
                assert tracker2.baseline_benchmark_time == 4.13
                assert tracker2.calibrated_benchmark_diff == pytest.approx(
                    (4.56 - 4.13) / 4.13, rel=1e-6
                )

                assert result == [
                    "display_str",
                    "display_str",
                    "display_str",
                    "display_str",
                ]  # Adjust this based on the actual expected result
                assert len(result) == 4  # Update based on expected result length


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
