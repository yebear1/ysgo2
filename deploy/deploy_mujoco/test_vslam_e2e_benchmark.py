#!/usr/bin/env python3
"""Deterministic checks for the VSLAM benchmark state machine."""

import tempfile
from pathlib import Path

import numpy as np
import yaml

from vslam_e2e_benchmark import VslamE2EBenchmark


class BridgeStub:
    map_updates = 0
    loop_closures = 0
    tracking_mode = "TEST"


def make_benchmark(tmpdir):
    config = {
        "mapping": {
            "timeout": 60.0,
            "waypoint_tolerance": 0.2,
            "waypoint_progress_step": 0.1,
            "waypoint_progress_timeout": 2.0,
            "waypoints": [[1.0, 0.0]],
            "initial_tracking_grace": 1.0,
            "initial_tracking_timeout": 4.0,
            "tracking_scan_delay": 0.2,
            "tracking_recovery_timeout": 2.0,
            "tracking_scan_yaw_rate": 0.3,
            "tracking_scan_period": 0.5,
        },
        "localization": {"timeout": 60.0, "goals": []},
        "thresholds": {},
    }
    path = Path(tmpdir) / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return VslamE2EBenchmark(path, "mapping", Path(tmpdir) / "result.json")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = make_benchmark(tmpdir)
        assert np.allclose(benchmark.tracking_recovery_command(None, 0.0), 0.0)
        assert np.allclose(benchmark.tracking_recovery_command(None, 0.5), 0.0)
        scan = benchmark.tracking_recovery_command(None, 1.1)
        assert np.allclose(scan[:2], 0.0) and abs(scan[2]) == 0.3

        pose = np.array([0.0, 0.0, 0.0])
        assert benchmark.tracking_recovery_command(pose, 1.2) is None
        assert benchmark.tracking_recovery_command(None, 1.3) is not None
        scan = benchmark.tracking_recovery_command(None, 1.6)
        assert abs(scan[2]) == 0.3
        assert benchmark.tracking_recovery_command(pose, 1.7) is None
        assert not benchmark.finished

        # A later unrecovered loss is bounded instead of hanging the suite.
        benchmark.tracking_recovery_command(None, 2.0)
        benchmark.tracking_recovery_command(None, 4.1)
        assert benchmark.finished
        assert benchmark.failed_reason == "visual tracking recovery timeout"

    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = make_benchmark(tmpdir)
        pose = np.array([0.0, 0.0, 0.0])
        benchmark.update_control(pose, None, 0.0)
        benchmark.update_control(pose, None, 2.1)
        assert benchmark.finished
        assert benchmark.failed_reason.startswith(
            "mapping waypoint progress timeout"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = make_benchmark(tmpdir)
        benchmark.phase_config["max_tracking_loss_events"] = 1
        pose = np.array([0.0, 0.0, 0.0])
        benchmark.tracking_recovery_command(pose, 0.0)
        benchmark.tracking_recovery_command(None, 0.1)
        benchmark.tracking_recovery_command(None, 0.4)
        benchmark.tracking_recovery_command(pose, 0.5)
        benchmark.tracking_recovery_command(None, 0.6)
        benchmark.tracking_recovery_command(None, 0.9)
        assert benchmark.finished
        assert benchmark.failed_reason == "repeated visual tracking loss: 2 events"

    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = make_benchmark(tmpdir)
        pose = np.array([0.0, 0.0, 0.0])
        benchmark.observe(0.0, pose, pose, None, BridgeStub())
        benchmark.observe(1.0, pose, None, None, BridgeStub())
        benchmark.observe(4.5, pose, None, None, BridgeStub())
        result = benchmark.result(BridgeStub())
        assert result["longest_tracking_loss_s"] == 3.5

    source = Path(__file__).with_name("vslam_e2e_benchmark.py").read_text()
    control_source = source[
        source.index("    def update_control") : source.index("    def observe")
    ]
    assert "last_truth" not in control_source
    assert "initial_truth" not in control_source
    deploy_source = Path(__file__).with_name("deploy_go2.py").read_text()
    yaw_block = deploy_source[
        deploy_source.index("                navigation_yaw = (") :
        deploy_source.index("                cmd = navigator.update(")
    ]
    assert "ros2_vslam.sensor_yaw" in yaw_block
    print("tracking_recovery=BOUNDED_ACTIVE_SCAN")
    print("truth_dependency_in_control=NONE")


if __name__ == "__main__":
    main()
