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


def main():
    check_scene("wall_avoidance_codex.xml", 0.25, "OBSTACLE")
    check_scene("cliff_avoidance_codex.xml", 0.45, "DROP")
    check_home_narrow_passage()


if __name__ == "__main__":
    main()
