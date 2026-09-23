#!/usr/bin/env python3
"""Regression checks for ground-truth-free global SLAM navigation."""

from pathlib import Path

import numpy as np

from goal_navigator import GoalNavigator


def make_grid():
    resolution = 0.10
    width = 60
    height = 50
    data = np.zeros((height, width), dtype=np.int8)
    # A wall across the direct route has a 0.8 m opening above the centre.
    wall_x = 30
    data[:, wall_x : wall_x + 2] = 100
    data[29:38, wall_x : wall_x + 2] = 0
    return {
        "update": 1,
        "resolution": resolution,
        "width": width,
        "height": height,
        "origin": (-3.0, -2.5),
        "data": data,
    }


def main():
    navigator = GoalNavigator(
        None,
        None,
        {
            "enabled": True,
            "map_source": "rtabmap",
            "robot_radius": 0.20,
            "preferred_clearance": 0.20,
            "smoothing_clearance": 0.20,
        },
    )
    assert not navigator.map_ready
    assert navigator.update_occupancy_grid(make_grid())
    assert navigator.map_ready

    start = np.array([-2.0, 0.0])
    goal = np.array([2.0, 0.0])
    assert navigator.set_goal(goal, start)
    assert len(navigator.path) >= 3
    waypoint_count = len(navigator.path)
    assert max(point[1] for point in navigator.path) > 0.20
    for point in navigator.path:
        assert not navigator.occupied[navigator._world_to_grid(point)]

    # Unknown RTAB-Map cells are deliberately non-navigable.
    unknown = make_grid()
    unknown["data"][:, 45:] = -1
    navigator.update_occupancy_grid(unknown)
    assert not navigator.set_goal(goal, start)

    bridge_source = (
        Path(__file__).with_name("ros2_vslam_bridge.py").read_text()
    )
    forbidden = ("_ground_truth_pose", "data.xpos", "data.xmat", "data.qpos")
    assert not any(token in bridge_source for token in forbidden)
    print(f"optimized_map_path_waypoints={waypoint_count}")
    print("unknown_space=NON_NAVIGABLE")
    print("sensor_bridge_ground_truth_dependency=NONE")


if __name__ == "__main__":
    main()
