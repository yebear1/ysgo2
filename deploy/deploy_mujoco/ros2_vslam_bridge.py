"""ROS 2 RGB-D sensor bridge for an external, graph-based VSLAM system.

This module deliberately does not read the simulated robot's world pose. It
publishes only camera measurements and calibration. RTAB-Map owns visual
odometry, loop closure, graph optimization and the global occupancy map.
"""

import math

import mujoco
import numpy as np


class Ros2VslamBridge:
    """Publish simulated sensors and consume RTAB-Map's optimized products."""

    def __init__(self, model, data, config):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped, TransformStamped
            from nav_msgs.msg import OccupancyGrid, Odometry
            from rclpy.duration import Duration
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from rclpy.time import Time
            from sensor_msgs.msg import CameraInfo, Image, Imu
            from std_msgs.msg import String
            from std_srvs.srv import Empty
            from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Launch with "
                "start_go2_vslam.sh after ROS 2 Jazzy is installed."
            ) from exc

        self._rclpy = rclpy
        self._Time = Time
        self._Duration = Duration
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._Imu = Imu
        self._String = String
        self._TransformStamped = TransformStamped
        self._Empty = Empty
        self.model = model
        self.data = data

        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 360))
        self.camera_name = str(config.get("camera_name", "front_rgbd"))
        self.frame_id = str(config.get("frame_id", "go2_camera_optical_frame"))
        self.base_frame_id = str(config.get("base_frame_id", "base_link"))
        self.map_frame_id = str(config.get("map_frame_id", "map"))
        self.publish_interval = float(config.get("publish_interval", 0.10))
        self.min_depth = float(config.get("min_depth", 0.12))
        self.max_depth = float(config.get("max_depth", 8.0))
        self.last_publish_time = -np.inf
        self.imu_publish_interval = float(config.get("imu_publish_interval", 0.01))
        self.last_imu_publish_time = -np.inf
        self._imu_last_simulation_time = None
        self._imu_orientation = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )

        self.tracking_mode = "WAITING FOR RTAB-MAP"
        self.odometry_messages = 0
        self.map_updates = 0
        self.loop_closures = 0
        self._last_loop_closure_id = 0
        self._odometry_lost = True
        self._global_pose = None
        self._latest_map = None
        self._consumed_map_update = 0
        self._navigation_goal = None

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("go2_mujoco_rgbd_sensor")
        self.rgb_pub = self.node.create_publisher(
            Image, "/go2/camera/color/image_raw", qos_profile_sensor_data
        )
        self.depth_pub = self.node.create_publisher(
            Image, "/go2/camera/depth/image_raw", qos_profile_sensor_data
        )
        self.info_pub = self.node.create_publisher(
            CameraInfo,
            "/go2/camera/color/camera_info",
            qos_profile_sensor_data,
        )
        self.imu_pub = self.node.create_publisher(
            Imu, "/go2/imu/data", qos_profile_sensor_data
        )
        self.status_pub = self.node.create_publisher(
            String, "/go2/vslam/status", 10
        )
        self.reset_odom_client = self.node.create_client(
            Empty, "/rtabmap/reset_odom"
        )
        # RTAB-Map's visual odometry uses sensor-data (best-effort) QoS.
        # Matching it is essential: a default reliable subscription is not
        # compatible with that publisher and silently receives no poses.
        self.node.create_subscription(
            Odometry,
            "/rtabmap/odom",
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            OccupancyGrid, "/map", self._map_callback, 1
        )
        self.node.create_subscription(
            PoseStamped, "/goal_pose", self._goal_callback, 10
        )

        # Keep the sensor bridge usable enough to report an installation error
        # even when rtabmap_msgs has not been installed yet.
        try:
            from rtabmap_msgs.msg import Info, OdomInfo

            self.node.create_subscription(
                Info, "/rtabmap/info", self._info_callback, 10
            )
            self.node.create_subscription(
                OdomInfo,
                "/rtabmap/odom_info",
                self._odom_info_callback,
                qos_profile_sensor_data,
            )
        except ImportError:
            pass

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(
            self.tf_buffer, self.node, spin_thread=False
        )
        self.static_broadcaster = StaticTransformBroadcaster(self.node)

        self.renderer = mujoco.Renderer(
            model, height=self.height, width=self.width
        )
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if camera_id < 0:
            raise RuntimeError(f"MuJoCo camera not found: {self.camera_name}")
        self.fovy = float(model.cam_fovy[camera_id])
        self._imu_gyro_slice = self._sensor_slice("imu_gyro")
        self._imu_accel_slice = self._sensor_slice("imu_accel")
        self.camera_info = self._build_camera_info()
        self._publish_static_camera_transform()
        print(
            "ROS 2 RGB-D sensor bridge (ground-truth-free): "
            f"{self.width}x{self.height} at "
            f"{1.0 / self.publish_interval:.1f} Hz"
        )

    def _sensor_slice(self, name):
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise RuntimeError(f"MuJoCo sensor not found: {name}")
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        if dimension != 3:
            raise RuntimeError(
                f"MuJoCo sensor '{name}' must have dimension 3, got {dimension}"
            )
        return slice(address, address + dimension)

    def _publish_static_camera_transform(self):
        transform = self._TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame_id
        transform.child_frame_id = self.frame_id
        transform.transform.translation.x = 0.30
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.04
        # Optical axes in base_link: +Z optical is +X base (forward), +X
        # optical is -Y base (right), and +Y optical is -Z base (down).
        transform.transform.rotation.w = 0.5
        transform.transform.rotation.x = -0.5
        transform.transform.rotation.y = 0.5
        transform.transform.rotation.z = -0.5
        self.static_broadcaster.sendTransform(transform)

    def _build_camera_info(self):
        info = self._CameraInfo()
        info.header.frame_id = self.frame_id
        info.width = self.width
        info.height = self.height
        fy = 0.5 * self.height / math.tan(math.radians(self.fovy) * 0.5)
        fx = fy
        cx = (self.width - 1.0) * 0.5
        cy = (self.height - 1.0) * 0.5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return info

    def _image_message(self, array, stamp, encoding):
        array = np.ascontiguousarray(array)
        message = self._Image()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.height = self.height
        message.width = self.width
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(array.strides[0])
        message.data = array.tobytes()
        return message

    def _publish_imu(self, simulation_time):
        """Publish gyro-integrated orientation without reading world pose."""
        simulation_time = float(simulation_time)
        previous_time = self._imu_last_simulation_time
        self._imu_last_simulation_time = simulation_time
        gyro = np.asarray(
            self.data.sensordata[self._imu_gyro_slice], dtype=np.float64
        ).copy()
        accel = np.asarray(
            self.data.sensordata[self._imu_accel_slice], dtype=np.float64
        ).copy()

        if previous_time is not None:
            dt = simulation_time - previous_time
            if 0.0 < dt <= 0.05:
                w, x, y, z = self._imu_orientation
                gx, gy, gz = gyro
                derivative = 0.5 * np.array(
                    [
                        -x * gx - y * gy - z * gz,
                        w * gx + y * gz - z * gy,
                        w * gy - x * gz + z * gx,
                        w * gz + x * gy - y * gx,
                    ],
                    dtype=np.float64,
                )
                self._imu_orientation += derivative * dt
                norm = float(np.linalg.norm(self._imu_orientation))
                if norm > 1.0e-9:
                    self._imu_orientation /= norm

        if simulation_time - self.last_imu_publish_time < self.imu_publish_interval:
            return
        self.last_imu_publish_time = simulation_time
        stamp = self.node.get_clock().now().to_msg()
        message = self._Imu()
        message.header.stamp = stamp
        message.header.frame_id = self.base_frame_id
        w, x, y, z = self._imu_orientation
        message.orientation.w = float(w)
        message.orientation.x = float(x)
        message.orientation.y = float(y)
        message.orientation.z = float(z)
        message.angular_velocity.x = float(gyro[0])
        message.angular_velocity.y = float(gyro[1])
        message.angular_velocity.z = float(gyro[2])
        message.linear_acceleration.x = float(accel[0])
        message.linear_acceleration.y = float(accel[1])
        message.linear_acceleration.z = float(accel[2])
        message.orientation_covariance = [
            0.0025, 0.0, 0.0,
            0.0, 0.0025, 0.0,
            0.0, 0.0, 0.0025,
        ]
        message.angular_velocity_covariance = [
            0.0004, 0.0, 0.0,
            0.0, 0.0004, 0.0,
            0.0, 0.0, 0.0004,
        ]
        message.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01,
        ]
        self.imu_pub.publish(message)

    def _odom_callback(self, message):
        self.odometry_messages += 1
        q = message.pose.pose.orientation
        quaternion_norm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        self._odometry_lost = quaternion_norm < 0.5
        if not self._odometry_lost and (
            self.tracking_mode.startswith("WAITING")
            or self.tracking_mode == "VISUAL TRACKING LOST"
        ):
            self.tracking_mode = "VISUAL ODOMETRY"

    def _odom_info_callback(self, message):
        # A null odometry message is still published while tracking is lost.
        # Invalidate the cached map pose immediately so navigation can never
        # continue on a stale transform.
        if bool(message.lost):
            self._odometry_lost = True
            self._global_pose = None
            self.tracking_mode = "VISUAL TRACKING LOST"

    def _map_callback(self, message):
        width = int(message.info.width)
        height = int(message.info.height)
        if width <= 0 or height <= 0 or len(message.data) != width * height:
            return
        self.map_updates += 1
        self._latest_map = {
            "update": self.map_updates,
            "resolution": float(message.info.resolution),
            "width": width,
            "height": height,
            "origin": (
                float(message.info.origin.position.x),
                float(message.info.origin.position.y),
            ),
            "data": np.asarray(message.data, dtype=np.int8)
            .reshape(height, width)
            .copy(),
        }

    def _info_callback(self, message):
        loop_id = int(getattr(message, "loop_closure_id", 0))
        proximity_id = int(getattr(message, "proximity_detection_id", 0))
        accepted_id = loop_id if loop_id > 0 else proximity_id
        if accepted_id > 0 and accepted_id != self._last_loop_closure_id:
            self.loop_closures += 1
            self._last_loop_closure_id = accepted_id
            self.tracking_mode = f"LOOP CLOSED #{self.loop_closures}"

    def _goal_callback(self, message):
        if message.header.frame_id not in ("", self.map_frame_id):
            self.node.get_logger().warning(
                f"Ignoring goal in '{message.header.frame_id}'; expected "
                f"'{self.map_frame_id}'"
            )
            return
        self._navigation_goal = np.array(
            [message.pose.position.x, message.pose.position.y],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_from_quaternion(rotation):
        return math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )

    def _update_global_pose(self):
        if self._odometry_lost:
            self._global_pose = None
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame_id,
                self.base_frame_id,
                self._Time(),
                timeout=self._Duration(seconds=0.0),
            )
        except Exception:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self._global_pose = np.array(
            [translation.x, translation.y, self._yaw_from_quaternion(rotation)],
            dtype=np.float64,
        )
        if not self.tracking_mode.startswith("LOOP CLOSED"):
            self.tracking_mode = "GLOBAL TRACKING"

    @property
    def global_pose(self):
        """Latest graph-optimized ``[x, y, yaw]`` pose, or ``None``."""
        return None if self._global_pose is None else self._global_pose.copy()

    def take_navigation_map(self):
        """Return each optimized occupancy-grid revision exactly once."""
        if (
            self._latest_map is None
            or self._latest_map["update"] == self._consumed_map_update
        ):
            return None
        self._consumed_map_update = self._latest_map["update"]
        result = dict(self._latest_map)
        result["data"] = self._latest_map["data"].copy()
        return result

    def take_navigation_goal(self):
        """Return and consume the latest RViz global goal."""
        if self._navigation_goal is None:
            return None
        goal = self._navigation_goal.copy()
        self._navigation_goal = None
        return goal

    def update(self, simulation_time):
        # Always service subscriptions so graph corrections are available to
        # the controller even between camera frames.
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        self._update_global_pose()
        self._publish_imu(simulation_time)
        if simulation_time - self.last_publish_time < self.publish_interval:
            return
        self.last_publish_time = float(simulation_time)

        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_name)
        depth = self.renderer.render().astype(np.float32, copy=True)
        self.renderer.disable_depth_rendering()

        # MuJoCo represents pixels that see the far clipping plane as a very
        # large metric depth (hundreds of metres), not as an invalid sample.
        # Feeding those values to RGB-D PnP makes distant/background features
        # look geometrically valid and quickly destabilizes visual odometry.
        # Real RGB-D cameras report those pixels as zero, so emulate that here.
        valid_depth = np.isfinite(depth)
        valid_depth &= depth >= self.min_depth
        valid_depth &= depth <= self.max_depth
        depth[~valid_depth] = 0.0

        stamp = self.node.get_clock().now().to_msg()
        self.camera_info.header.stamp = stamp
        self.rgb_pub.publish(self._image_message(rgb, stamp, "rgb8"))
        self.depth_pub.publish(self._image_message(depth, stamp, "32FC1"))
        self.info_pub.publish(self.camera_info)

        status = self._String()
        pose_status = "pose=ready" if self._global_pose is not None else "pose=waiting"
        status.data = (
            f"{self.tracking_mode}; {pose_status}; "
            f"odom={self.odometry_messages}; maps={self.map_updates}; "
            f"loops={self.loop_closures}"
        )
        self.status_pub.publish(status)

    def reset(self):
        # A simulator reset is deliberately not allowed to erase the external
        # RTAB-Map database. Reset only local visual odometry; the global graph
        # then relocalizes the camera against the persistent map.
        if self.reset_odom_client.service_is_ready():
            self.reset_odom_client.call_async(self._Empty.Request())
        self._global_pose = None
        self._odometry_lost = True
        self.tracking_mode = "WAITING FOR RELOCALIZATION"
        self.last_publish_time = -np.inf
        self.last_imu_publish_time = -np.inf
        self._imu_last_simulation_time = None
        self._imu_orientation[:] = (1.0, 0.0, 0.0, 0.0)

    def close(self):
        self.renderer.close()
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
