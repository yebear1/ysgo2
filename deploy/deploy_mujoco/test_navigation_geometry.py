#!/usr/bin/env python3
"""Check wall and cliff detection against real MuJoCo ray-cast geometry."""

from pathlib import Path

import mujoco
import numpy as np
import yaml

from lidar_heightmap import LidarHeightMap
from terrain_navigator import TerrainNavigator


ROOT = Path(__file__).resolve().parents[2]


def check_scene(xml_name, robot_x, expected):
    config = yaml.safe_load(
        (ROOT / "deploy/deploy_mujoco/configs/go2_stairs.yaml").read_text()
    )
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "resources/robots/go2" / xml_name)
    )
    data = mujoco.MjData(model)
    data.qpos[0] = robot_x
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, config["lidar"])
    lidar.scan()
    navigator = TerrainNavigator(config["navigation"])
    result = navigator.analyze(lidar)
    command = navigator.update(np.array([0.8, 0.0, 0.0]), lidar, 0.0)
    assert result[0] == expected, (xml_name, result)
    assert command[0] == 0.0 and abs(command[2]) > 0.0, command
    print(
        f"{xml_name}: {result[0]} at {result[1]:.2f}m, "
        f"target={np.degrees(navigator.target_relative_angle):+.0f}deg, "
        f"command={command.tolist()}"
    )


def check_home_narrow_passage():
    """The real armchair passage has a safe shallow arc despite no pivot room."""
    config = yaml.safe_load(
        (ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text()
    )
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "resources/robots/go2/home_codex.xml")
    )
    data = mujoco.MjData(model)
    data.qpos[:7] = [
        3.1653237281,
        2.7548543275,
        0.2825403209,
        0.7507599138,
        -0.0080767746,
        0.0091810284,
        0.6604619794,
    ]
    data.qpos[7:19] = config["default_angles"]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, config["lidar"])
    lidar.scan()
    navigator = TerrainNavigator(config["navigation"])
    navigator.state = "BLOCKED"
    command = navigator.update(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        lidar,
        np.deg2rad(82.6777678),
    )
    assert navigator.state == "STEER_FORWARD", navigator.status_text()
    assert command[0] > 0.0 and command[2] > 0.0, command
    assert abs(np.degrees(navigator.target_relative_angle)) <= 15.1
    print(
        "home narrow passage: "
        f"target={np.degrees(navigator.target_relative_angle):+.0f}deg, "
        f"command={command.tolist()}"
    )


def check_home_side_wall_escape():
    """A turn beside one wall must step into open space instead of deadlocking."""
    config = yaml.safe_load(
        (ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text()
    )
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "resources/robots/go2/home_codex.xml")
    )
    data = mujoco.MjData(model)
    data.qpos[:7] = [
        -1.9701971452,
        5.6800896052,
        0.2825403203,
        0.9999206884,
        0.0000368758,
        0.0122280103,
        -0.0030151701,
    ]
    data.qpos[7:19] = config["default_angles"]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, config["lidar"])
    lidar.scan()
    navigator = TerrainNavigator(config["navigation"])
    turn_right = np.array([0.0, 0.0, -1.2], dtype=np.float32)
    navigator.state = "BLOCKED"
    command = navigator.update(turn_right, lidar, np.deg2rad(-0.3455))
    assert navigator.state == "LATERAL_FOR_TURN", navigator.status_text()
    assert command[0] == 0.0 and command[1] < 0.0 and command[2] == 0.0

    # Moving right (negative body Y at this heading) opens the measured turn
    # envelope; the original right-turn command must then resume.
    data.qpos[1] -= 0.22
    mujoco.mj_forward(model, data)
    lidar.scan()
    resumed = navigator.update(turn_right, lidar, np.deg2rad(-0.3455))
    assert navigator.state == "FORWARD", navigator.status_text()
    assert np.allclose(resumed, turn_right)
    print(
        "home side-wall escape: "
        f"command={command.tolist()}, resumed={resumed.tolist()}"
    )


def main():
    check_scene("wall_avoidance_codex.xml", 0.25, "OBSTACLE")
    check_scene("cliff_avoidance_codex.xml", 0.45, "DROP")
    check_home_narrow_passage()
    check_home_side_wall_escape()
    check_low_object_scan()
    check_toy_between_elevation_samples()
    check_low_scan_above_rug()
    check_recorded_rear_foot_arc()


def check_low_object_scan():
    config = yaml.safe_load((ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/go2/home_codex.xml"))
    data = mujoco.MjData(model)
    data.qpos[:7] = [-2.0, 1.75, 0.30, 1, 0, 0, 0]
    data.qpos[7:19] = config["default_angles"]
    mujoco.mj_forward(model, data)
    upper = LidarHeightMap(model, data, config["lidar"])
    lower = LidarHeightMap(model, data, dict(config["lidar"], planar_scan_heights=[0.05, 0.32]))
    upper.scan()
    lower.scan()
    index = np.argmin(np.abs(lower.planar_angles))
    assert lower.planar_ranges[index] < 0.39, lower.planar_ranges[index]
    assert upper.planar_ranges[index] > 0.60, upper.planar_ranges[index]
    navigator = TerrainNavigator(config["navigation"])
    command = navigator.update([0, 0, 0.45], lower, 0, autonomous=True)
    assert command[2] == 0.0 and np.linalg.norm(command[:2]) > 0.0, command
    indoor = TerrainNavigator(dict(config["navigation"], max_step_up=0.04))
    assert indoor.analyze(lower)[0] == "OBSTACLE"
    print(f"low book: range={lower.planar_ranges[index]:.3f}m; safe escape={command}")


def check_toy_between_elevation_samples():
    """Recorded pre-contact pose: toy is inside footprint but between cells."""
    config = yaml.safe_load((ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/go2/home_codex.xml"))
    data = mujoco.MjData(model)
    yaw = -0.12
    data.qpos[:7] = [0.94, 1.48, 0.30, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qpos[7:19] = config["default_angles"]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, dict(config["lidar"], planar_scan_heights=[0.05, 0.32]))
    lidar.scan()
    legacy = TerrainNavigator(dict(config["navigation"], max_step_up=0.04))
    fixed = TerrainNavigator(dict(config["navigation"], max_step_up=0.04, planar_obstacle_check=True))
    assert legacy.analyze(lidar)[0] == "CLEAR"
    hazard, distance = fixed.analyze(lidar)
    assert hazard == "OBSTACLE" and 0.45 < distance < 0.60, (hazard, distance)
    assert fixed._candidate_result(lidar, 0.0)[0] == "OBSTACLE"
    # A measured return outside the exact footprint is not an invisible wall.
    lidar.elevation.fill(0)
    lidar.planar_ranges.fill(lidar.planar_max_range)
    index = np.argmin(np.abs(lidar.planar_angles - np.pi / 4))
    lidar.planar_ranges[index] = 0.50
    assert fixed.analyze(lidar)[0] == "CLEAR"
    print(f"toy between grid samples: detected at {distance:.3f}m, no footprint inflation")


def check_low_scan_above_rug():
    config = yaml.safe_load((ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/go2/home_codex.xml"))
    data = mujoco.MjData(model)
    yaw = 0.520141
    data.qpos[:7] = [-1.408105, 1.3567, 0.30, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qpos[7:19] = config["default_angles"]
    mujoco.mj_forward(model, data)
    old = LidarHeightMap(model, data, dict(config["lidar"], planar_scan_heights=[0.05, 0.32]))
    fixed = LidarHeightMap(model, data, dict(config["lidar"], planar_scan_heights=config["ros2_vslam"]["planar_scan_heights"]))
    old.scan()
    fixed.scan()
    assert abs(fixed.baseline - 0.024) < 1e-6, fixed.baseline
    assert old.planar_clearance() > 0.5
    assert fixed.planar_clearance() < 0.3
    print(f"book beside rug: old range={old.planar_clearance():.3f}, corrected={fixed.planar_clearance():.3f}")


def check_recorded_rear_foot_arc():
    """Reconstruct the measured failing gait pose; truth is test-only input."""
    config = yaml.safe_load((ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/go2/home_codex.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = [
        -1.6862648063, 1.3874275675, 0.3482281996,
        0.8683298324, -0.0008938442, 0.0107929500, -0.4958689498,
        0.0754702744, 0.9568666374, -1.0161424751,
        0.1317224087, 0.7716578586, -1.6169130900,
        0.1332249653, 0.8552972357, -1.5120527947,
        -0.0867419629, 0.7440004556, -1.2476749376,
    ]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, dict(config["lidar"], planar_scan_heights=[0.04, 0.32]))
    lidar.scan()
    navigator = TerrainNavigator(dict(config["navigation"], max_step_up=0.04, planar_obstacle_check=True))
    old_command = [0.2104455084, 0, -0.3675268292]
    assert not navigator._swept_motion_clear(old_command, lidar)
    corrected = navigator._guard_swept_motion(old_command, lidar)
    assert corrected[0] > 0 and abs(corrected[2]) < 1e-6
    assert navigator._swept_motion_clear(corrected, lidar)
    print(f"recorded rear-foot contact: unsafe arc replaced by {corrected}")


if __name__ == "__main__":
    main()
