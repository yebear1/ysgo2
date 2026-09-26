"""Pose freshness and graph-correction regressions; no ROS daemon required."""

import math
import unittest
from types import SimpleNamespace as NS

import numpy as np

from ros2_vslam_bridge import Ros2VslamBridge


def stamp(seconds):
    ns = round(seconds * 1e9)
    return NS(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def transform(seconds, yaw):
    return NS(
        header=NS(stamp=stamp(seconds)),
        transform=NS(
            translation=NS(x=1.0, y=2.0),
            rotation=NS(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
        ),
    )


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = Ros2VslamBridge.__new__(Ros2VslamBridge)
        b = self.bridge
        b._pending_map_correction = None
        b._map_match_not_before_ns = 0
        b._verified_map_match = False
        b._sensor_time_ns = 10_000_000_000
        b._last_odom_stamp_ns = 9_900_000_000
        b._odometry_lost = False
        b.pose_timeout = 0.5
        b._global_pose = np.array([4., 5., 0.])
        b._imu_orientation = np.array([1., 0., 0., 0.])
        b._imu_map_yaw_offset = 0.0
        b.map_frame_id, b.base_frame_id = "map", "base_link"
        b._Time = lambda: None
        b._Duration = lambda **kwargs: None
        b.tracking_mode = "GLOBAL TRACKING"
        b.odometry_messages = 0
        b._navigation_velocity = np.zeros(3)
        self.tf = transform(9.9, 0.2)
        b.tf_buffer = NS(lookup_transform=lambda *args, **kwargs: self.tf)

    def test_graph_heading_correction_is_used(self):
        self.bridge._update_global_pose()
        self.assertAlmostEqual(self.bridge.global_pose[2], 0.2)
        self.tf = transform(10.0, 0.7)
        self.bridge._update_global_pose()
        self.assertAlmostEqual(self.bridge.global_pose[2], 0.7)

    def test_match_notification_waits_for_its_corrected_tf(self):
        b = self.bridge
        b.loop_closures = 0
        b._last_loop_closure_id = 0
        expected = transform(10, math.pi).transform
        expected.translation.z = 0.0
        old = transform(10, 0.0).transform
        old.translation.z = 0.0
        map_tf = NS(transform=old)
        b.tf_buffer = NS(lookup_transform=lambda target, source, *args, **kwargs:
                         map_tf if source == "odom" else self.tf)
        b._info_callback(NS(loop_closure_id=123, proximity_detection_id=0,
                            odom_cache=NS(map_to_odom=expected)))
        self.assertIsNone(b.global_pose)
        b._update_global_pose()
        self.assertIsNone(b.global_pose)  # Fresh but uncorrected TF is unsafe.
        self.assertFalse(b._verified_map_match)
        map_tf.transform = expected
        b._update_global_pose()
        self.assertIsNone(b.global_pose)
        b._sensor_time_ns += 500_000_000
        b._last_odom_stamp_ns = b._sensor_time_ns
        self.tf = transform(10.5, math.pi)
        b._update_global_pose()
        self.assertTrue(b._verified_map_match)
        self.assertIsNotNone(b.global_pose)

    def test_navigation_velocity_comes_from_visual_odometry(self):
        self.bridge._odom_callback(NS(
            header=NS(stamp=stamp(10.0)),
            pose=NS(pose=NS(orientation=NS(x=0., y=0., z=0., w=1.))),
            twist=NS(twist=NS(linear=NS(x=.3, y=-.1), angular=NS(z=.2))),
        ))
        np.testing.assert_allclose(self.bridge.navigation_velocity, [.3, -.1, .2])
        self.bridge._global_pose = None
        np.testing.assert_array_equal(self.bridge.navigation_velocity, [0, 0, 0])

    def test_old_odometry_cannot_drive(self):
        self.bridge._last_odom_stamp_ns = 9_000_000_000
        self.bridge._update_global_pose()
        self.assertIsNone(self.bridge.global_pose)

    def test_old_tf_cannot_drive_even_with_fresh_odometry(self):
        self.tf = transform(9.0, 0.1)
        self.bridge._update_global_pose()
        self.assertIsNone(self.bridge.global_pose)

    def test_tf_failure_clears_cached_pose(self):
        def unavailable(*args, **kwargs):
            raise RuntimeError("missing map frame")
        self.bridge.tf_buffer.lookup_transform = unavailable
        self.bridge._update_global_pose()
        self.assertIsNone(self.bridge.global_pose)

    def test_delayed_loss_does_not_override_new_success(self):
        self.bridge._odom_info_callback(NS(header=NS(stamp=stamp(9.0)), lost=True))
        self.assertFalse(self.bridge._odometry_lost)
        self.bridge._odom_info_callback(NS(
            header=NS(stamp=stamp(10.0)), lost=True, inliers=0, matches=3, features=20,
        ))
        self.assertTrue(self.bridge._odometry_lost)
        self.assertIsNone(self.bridge.global_pose)
        # A late pose from before the new loss cannot resurrect tracking.
        old_pose = NS(
            header=NS(stamp=stamp(9.95)),
            pose=NS(pose=NS(orientation=NS(x=0., y=0., z=0., w=1.))),
        )
        self.bridge._odom_callback(old_pose)
        self.assertTrue(self.bridge._odometry_lost)

    def test_atomic_rgbd_uses_capture_time_not_wall_time(self):
        b = self.bridge
        streams = {}
        for name in ("clock_pub", "rgbd_pub", "rgb_pub", "depth_pub", "info_pub", "status_pub"):
            streams[name] = []
            setattr(b, name, NS(publish=streams[name].append))
        b._Time = lambda nanoseconds: NS(to_msg=lambda: stamp(nanoseconds / 1e9))
        b._Clock = b._RGBDImage = b._String = NS
        b._Image = lambda: NS(header=NS())
        b._rclpy = NS(spin_once=lambda *args, **kwargs: None)
        b._update_global_pose = lambda: None
        b._publish_imu = lambda t: None
        b.node = b.model = b.data = None
        b.width = b.height = 2
        b.frame_id, b.camera_name = "optical", "front_rgbd"
        b.min_depth, b.max_depth = 0.12, 8.0
        b.last_imu_publish_time = b.last_publish_time = -math.inf
        b.imu_publish_interval, b.publish_interval = 0.01, 0.1
        b.camera_blocked = b._capture_loss = False
        b.camera_frames = b.map_updates = b.loop_closures = 0
        b.camera_info = NS(header=NS(stamp=None))
        depth_mode = [False]
        b.renderer = NS(
            disable_depth_rendering=lambda: depth_mode.__setitem__(0, False),
            enable_depth_rendering=lambda: depth_mode.__setitem__(0, True),
            update_scene=lambda *args, **kwargs: None,
            render=lambda: np.ones((2, 2), np.float32) if depth_mode[0] else np.zeros((2, 2, 3), np.uint8),
        )
        b.update(12.5)
        sample = streams["rgbd_pub"][0]
        for message in (sample, sample.rgb, sample.depth, sample.rgb_camera_info, sample.depth_camera_info):
            self.assertEqual(b._stamp_ns(message.header.stamp), 12_500_000_000)
        self.assertEqual(sample.rgb.encoding, "rgb8")
        self.assertEqual(sample.depth.encoding, "32FC1")
        self.assertEqual(b.camera_frames, 1)


if __name__ == "__main__":
    unittest.main()
