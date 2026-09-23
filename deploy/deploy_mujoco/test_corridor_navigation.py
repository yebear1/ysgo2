#!/usr/bin/env python3
"""Regression checks for indoor wall perception and symmetric corridor control.

Run directly with a Python environment containing MuJoCo, NumPy and OpenCV.
"""
import math
import unittest

import mujoco
import numpy as np

from lidar_heightmap import LidarHeightMap
from terrain_navigator import TerrainNavigator


def corridor(offset=0.0, yaw=0.0, filter_obstacles=True, right_wall=True):
    right = '<geom name="right" type="box" group="1" pos="0 -.36 .5" size="5 .05 .5"/>' if right_wall else ''
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="floor" type="plane" group="1" size="0 0 .05"/>
      <geom name="left" type="box" group="1" pos="0 .36 .5" size="5 .05 .5"/>
      {right}
      <body name="base" pos="0 0 .35"><freejoint/>
        <geom type="sphere" size=".02" mass="1" group="2" contype="0" conaffinity="0"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[1] = offset
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, {"policy_filter_obstacles": filter_obstacles})
    lidar.scan()
    return lidar


class CorridorNavigationTests(unittest.TestCase):
    def test_wall_filter_does_not_hide_obstacles_from_navigation(self):
        raw = corridor(filter_obstacles=False)
        filtered = corridor()
        self.assertGreater(float(raw.policy_heights.max()), .9)
        np.testing.assert_allclose(filtered.policy_heights, 0.0, atol=1e-6)
        np.testing.assert_allclose(filtered.elevation, raw.elevation)
        np.testing.assert_allclose(filtered.planar_ranges, raw.planar_ranges)
        self.assertAlmostEqual(filtered.planar_clearance(math.pi / 2, .2), .31, places=5)
        # The local planner still sees a wall when evaluating a turn into it.
        self.assertEqual(TerrainNavigator({})._candidate_result(filtered, math.pi / 2)[0], "OBSTACLE")

    def test_connected_stairs_and_negative_drops_are_preserved(self):
        lidar = corridor()
        stairs = np.maximum(0, np.floor((lidar.policy_x_values + 1e-6) / .2)) * .12
        lidar.policy_heights[:] = stairs[:, None]
        before = lidar.policy_heights.copy()
        lidar._filter_policy_obstacles()
        np.testing.assert_array_equal(lidar.policy_heights, before)
        self.assertGreater(float(before.max()), .4)
        lidar.policy_heights[:] = lidar.policy_x_values[:, None] * .5
        before = lidar.policy_heights.copy()
        lidar._filter_policy_obstacles()
        np.testing.assert_array_equal(lidar.policy_heights, before)
        lidar.policy_heights[:] = 0.0
        lidar.policy_heights[lidar.policy_x_values >= .4] = -.4
        before = lidar.policy_heights.copy()
        lidar._filter_policy_obstacles()
        np.testing.assert_array_equal(lidar.policy_heights, before)

    def test_skewed_body_recovers_symmetrically(self):
        commands = []
        for side in (-1, 1):
            lidar = corridor(offset=side * .04, yaw=side * .08)
            offset, heading, width = lidar.corridor_alignment()
            self.assertAlmostEqual(offset, -side * .04, places=5)
            self.assertAlmostEqual(heading, -side * .08, places=5)
            self.assertAlmostEqual(width, .62, places=5)
            navigator = TerrainNavigator({})
            command = navigator.update(
                [.75, 0, 0], lidar, side * .08, lateral_velocity=side * .05
            )
            self.assertEqual(navigator.state, "FORWARD")
            self.assertLess(side * command[1], 0)
            self.assertLess(side * command[2], 0)
            self.assertLessEqual(command[0], .35 + 1e-6)
            commands.append(command)
        np.testing.assert_allclose(commands[0], commands[1] * [1, -1, -1], atol=1e-6)

    def test_open_space_and_single_wall_do_not_force_corridor_heading(self):
        lidar = corridor(right_wall=False)
        self.assertIsNone(lidar.corridor_alignment())
        navigator = TerrainNavigator({})
        command = navigator.update([.5, 0, -.1], lidar, 0)
        self.assertAlmostEqual(float(command[2]), -.1, places=6)
        lidar.planar_ranges[:] = lidar.planar_max_range
        self.assertIsNone(lidar.corridor_alignment())

    def test_learned_lateral_and_yaw_commands_still_center_in_corridor(self):
        # PPO's nonzero Vy and yaw used to bypass all centring, driving a
        # physically passable corridor into the furniture on one side.
        for side in (-1, 1):
            lidar = corridor(offset=side * .04, yaw=side * .08)
            navigator = TerrainNavigator({})
            command = navigator.update(
                [.75, side * .2, side * .8], lidar, side * .08,
                autonomous=True, lateral_velocity=side * .05,
            )
            self.assertLess(side * command[1], 0)
            self.assertLess(side * command[2], 0)
            self.assertLessEqual(command[0], .35 + 1e-6)
        # At the centre, a learned lateral command must not survive merely
        # because the correction happens to be exactly zero.
        command = TerrainNavigator({}).update(
            [.75, .2, .8], corridor(), 0, autonomous=True,
        )
        np.testing.assert_allclose(command, [.35, 0, 0], atol=1e-6)

    def test_manual_lateral_command_is_preserved(self):
        request = [.5, .2, .1]
        command = TerrainNavigator({}).update(request, corridor(), 0)
        np.testing.assert_allclose(command, request)

    def test_selected_avoidance_heading_and_stop_are_preserved(self):
        lidar = corridor()
        navigator = TerrainNavigator({})
        navigator.state = "STEER_FORWARD"
        navigator.target_yaw = .17
        command = navigator.update([.5, 0, 0], lidar, 0)
        self.assertGreater(command[2], .2)
        np.testing.assert_array_equal(navigator.update([0, 0, 0], lidar, 0), [0, 0, 0])
        self.assertEqual(navigator.state, "FORWARD")


if __name__ == "__main__":
    unittest.main()
