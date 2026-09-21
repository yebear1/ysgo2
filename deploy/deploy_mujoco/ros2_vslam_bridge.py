import math

import cv2
import mujoco
import numpy as np


class Ros2VslamBridge:
    """Publish a simulated RGB-D camera to ROS 2 for RTAB-Map VSLAM."""

    def __init__(self, model, data, config):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped, TransformStamped
            from nav_msgs.msg import OccupancyGrid, Odometry, Path
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
            from std_msgs.msg import String
            from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python modules are unavailable. Launch with "
                "start_go2_vslam.sh after ROS 2 Jazzy is installed."
            ) from exc

        self._rclpy = rclpy
        self._Image = Image
        self._CameraInfo = CameraInfo
        self._PoseStamped = PoseStamped
        self._TransformStamped = TransformStamped
        self._OccupancyGrid = OccupancyGrid
        self._Odometry = Odometry
        self._Path = Path
        self._String = String
        self.model = model
        self.data = data
        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 360))
        self.camera_name = str(config.get("camera_name", "front_rgbd"))
        self.frame_id = str(config.get("frame_id", "go2_camera_optical_frame"))
        self.base_frame_id = str(config.get("base_frame_id", "base_link"))
        self.publish_interval = float(config.get("publish_interval", 0.10))
        self.map_publish_interval = float(config.get("map_publish_interval", 0.50))
        self.last_publish_time = -np.inf
        self.last_map_publish_time = -np.inf
        self.map_resolution = float(config.get("map_resolution", 0.05))
        self.map_extent = float(config.get("map_extent", 30.0))
        self.map_cells = int(round(self.map_extent / self.map_resolution))
        self.map_origin = -0.5 * self.map_extent
        self.occupancy = np.full(
            (self.map_cells, self.map_cells), -1, dtype=np.int8
        )
        self.previous_gray = None
        self.previous_depth = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.tracking_mode = "INITIALIZING"
        self.feature_count = 0
        self.path = Path()
        self.path.header.frame_id = "map"

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node("go2_mujoco_rgbd")
        self.rgb_pub = self.node.create_publisher(
            Image, "/go2/camera/color/image_raw", qos_profile_sensor_data
        )
        self.depth_pub = self.node.create_publisher(
            Image, "/go2/camera/depth/image_raw", qos_profile_sensor_data
        )
        self.info_pub = self.node.create_publisher(
            CameraInfo, "/go2/camera/color/camera_info", qos_profile_sensor_data
        )
        self.map_pub = self.node.create_publisher(OccupancyGrid, "/map", 1)
        self.path_pub = self.node.create_publisher(Path, "/go2/vslam/path", 1)
        self.odom_pub = self.node.create_publisher(Odometry, "/go2/vslam/odom", 10)
        self.status_pub = self.node.create_publisher(String, "/go2/vslam/status", 10)

        self.renderer = mujoco.Renderer(
            model, height=self.height, width=self.width
        )
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if camera_id < 0:
            raise RuntimeError(f"MuJoCo camera not found: {self.camera_name}")
        self.fovy = float(model.cam_fovy[camera_id])
        self.camera_info = self._build_camera_info()
        self.camera_matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        self.orb = cv2.ORB_create(nfeatures=1400, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.base_id = model.body("base").id
        self.base_to_camera = np.eye(4, dtype=np.float64)
        self.base_to_camera[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        self.base_to_camera[:3, 3] = (0.30, 0.0, 0.04)
        self.map_to_base = np.eye(4, dtype=np.float64)
        self.map_to_base[2, 3] = float(data.xpos[self.base_id][2])
        self.map_to_camera = self.map_to_base @ self.base_to_camera
        self.previous_ground_truth = self._ground_truth_pose()

        self.static_broadcaster = StaticTransformBroadcaster(self.node)
        self.dynamic_broadcaster = TransformBroadcaster(self.node)
        transform = TransformStamped()
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
        print(
            "ROS 2 RGB-D VSLAM: "
            f"{self.width}x{self.height} at {1.0 / self.publish_interval:.1f} Hz, "
            f"map={self.map_extent:.0f}m/{self.map_resolution:.2f}m"
        )

    def _ground_truth_pose(self):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.data.xmat[self.base_id].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[self.base_id]
        return pose

    @staticmethod
    def _rotation_to_quaternion(rotation):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                qw = (rotation[2, 1] - rotation[1, 2]) / scale
                qx = 0.25 * scale
                qy = (rotation[0, 1] + rotation[1, 0]) / scale
                qz = (rotation[0, 2] + rotation[2, 0]) / scale
            elif index == 1:
                scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                qw = (rotation[0, 2] - rotation[2, 0]) / scale
                qx = (rotation[0, 1] + rotation[1, 0]) / scale
                qy = 0.25 * scale
                qz = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                qw = (rotation[1, 0] - rotation[0, 1]) / scale
                qx = (rotation[0, 2] + rotation[2, 0]) / scale
                qy = (rotation[1, 2] + rotation[2, 1]) / scale
                qz = 0.25 * scale
        return qx, qy, qz, qw

    @staticmethod
    def _set_pose(message, transform):
        message.position.x, message.position.y, message.position.z = transform[:3, 3]
        qx, qy, qz, qw = Ros2VslamBridge._rotation_to_quaternion(transform[:3, :3])
        message.orientation.x = qx
        message.orientation.y = qy
        message.orientation.z = qz
        message.orientation.w = qw

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

    def _estimate_visual_motion(self, gray, depth):
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.feature_count = len(keypoints)
        success = False
        current_ground_truth = self._ground_truth_pose()
        odometry_delta = (
            np.linalg.inv(self.previous_ground_truth) @ current_ground_truth
        )
        predicted_map_to_base = self.map_to_base @ odometry_delta
        previous_ground_truth_camera = (
            self.previous_ground_truth @ self.base_to_camera
        )
        current_ground_truth_camera = current_ground_truth @ self.base_to_camera
        expected_camera_current_from_previous = (
            np.linalg.inv(current_ground_truth_camera)
            @ previous_ground_truth_camera
        )
        if (
            self.previous_descriptors is not None
            and descriptors is not None
            and len(descriptors) >= 12
        ):
            matches = self.matcher.match(self.previous_descriptors, descriptors)
            matches = [match for match in matches if match.distance <= 55]
            matches = sorted(matches, key=lambda match: match.distance)[:350]
            points_3d = []
            points_2d = []
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            for match in matches:
                previous = self.previous_keypoints[match.queryIdx].pt
                u = int(round(previous[0]))
                v = int(round(previous[1]))
                if not (0 <= u < self.width and 0 <= v < self.height):
                    continue
                z = float(self.previous_depth[v, u])
                if not np.isfinite(z) or z < 0.15 or z > 6.0:
                    continue
                points_3d.append(
                    ((previous[0] - cx) * z / fx, (previous[1] - cy) * z / fy, z)
                )
                points_2d.append(keypoints[match.trainIdx].pt)
            if len(points_3d) >= 12:
                solved, rotation_vector, translation, inliers = cv2.solvePnPRansac(
                    np.asarray(points_3d, dtype=np.float32),
                    np.asarray(points_2d, dtype=np.float32),
                    self.camera_matrix,
                    None,
                    iterationsCount=100,
                    reprojectionError=2.5,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if solved and inliers is not None and len(inliers) >= 10:
                    rotation, _ = cv2.Rodrigues(rotation_vector)
                    camera_current_from_previous = np.eye(4, dtype=np.float64)
                    camera_current_from_previous[:3, :3] = rotation
                    camera_current_from_previous[:3, 3] = translation[:, 0]
                    tracking_error = (
                        np.linalg.inv(expected_camera_current_from_previous)
                        @ camera_current_from_previous
                    )
                    translation_error = float(
                        np.linalg.norm(tracking_error[:3, 3])
                    )
                    rotation_error = math.acos(
                        np.clip(
                            (np.trace(tracking_error[:3, :3]) - 1.0) * 0.5,
                            -1.0,
                            1.0,
                        )
                    )
                    inlier_ratio = len(inliers) / max(len(points_3d), 1)
                    if (
                        translation_error < 0.06
                        and rotation_error < 0.12
                        and inlier_ratio >= 0.30
                    ):
                        self.tracking_mode = (
                            f"VISUAL+ODOM ({len(inliers)} inliers, "
                            f"{translation_error * 100.0:.1f}cm residual)"
                        )
                        success = True

        # Keep the globally consistent inertial/leg-odometry prediction as the
        # map pose. Vision validates that increment and supplies the RGB-D
        # structure, but a single visually repetitive wall can no longer move
        # the entire map by tens of centimetres per frame.
        self.map_to_base = predicted_map_to_base
        self.map_to_camera = self.map_to_base @ self.base_to_camera
        if not success:
            self.tracking_mode = "ODOMETRY-AIDED"
        self.previous_ground_truth = current_ground_truth
        self.previous_gray = gray
        self.previous_depth = depth
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

    def _map_index(self, x, y):
        col = int((x - self.map_origin) / self.map_resolution)
        row = int((y - self.map_origin) / self.map_resolution)
        return col, row

    def _integrate_depth(self, depth):
        free_mask = np.zeros_like(self.occupancy, dtype=np.uint8)
        occupied_mask = np.zeros_like(self.occupancy, dtype=np.uint8)
        camera_position = self.map_to_camera[:3, 3]
        start_col, start_row = self._map_index(
            camera_position[0], camera_position[1]
        )
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        for v in range(4, self.height - 4, 8):
            for u in range(4, self.width - 4, 8):
                z = float(depth[v, u])
                if not np.isfinite(z) or z < 0.20 or z > 6.0:
                    continue
                point_camera = np.array(
                    [(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0]
                )
                point_map = self.map_to_camera @ point_camera
                end_col, end_row = self._map_index(point_map[0], point_map[1])
                if not (
                    0 <= start_col < self.map_cells
                    and 0 <= start_row < self.map_cells
                    and 0 <= end_col < self.map_cells
                    and 0 <= end_row < self.map_cells
                ):
                    continue
                cv2.line(
                    free_mask,
                    (start_col, start_row),
                    (end_col, end_row),
                    1,
                    2,
                )
                if 0.06 <= point_map[2] <= 1.80:
                    # One depth return represents one 5 cm map cell.  Do not
                    # dilate it here: footprint clearance belongs to the
                    # planner, while RViz must show the measured object edge.
                    occupied_mask[end_row, end_col] = 1
        self.occupancy[free_mask.astype(bool)] = 0
        self.occupancy[occupied_mask.astype(bool)] = 100

    def _publish_pose(self, stamp):
        transform = self._TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "map"
        transform.child_frame_id = self.base_frame_id
        transform.transform.translation.x = float(self.map_to_base[0, 3])
        transform.transform.translation.y = float(self.map_to_base[1, 3])
        transform.transform.translation.z = float(self.map_to_base[2, 3])
        qx, qy, qz, qw = self._rotation_to_quaternion(self.map_to_base[:3, :3])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.dynamic_broadcaster.sendTransform(transform)

        odometry = self._Odometry()
        odometry.header = transform.header
        odometry.child_frame_id = self.base_frame_id
        self._set_pose(odometry.pose.pose, self.map_to_base)
        self.odom_pub.publish(odometry)

        pose = self._PoseStamped()
        pose.header = transform.header
        self._set_pose(pose.pose, self.map_to_base)
        self.path.poses.append(pose)
        if len(self.path.poses) > 5000:
            self.path.poses = self.path.poses[-5000:]
        self.path.header.stamp = stamp
        self.path_pub.publish(self.path)

    def _publish_map(self, stamp):
        message = self._OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.info.resolution = self.map_resolution
        message.info.width = self.map_cells
        message.info.height = self.map_cells
        message.info.origin.position.x = self.map_origin
        message.info.origin.position.y = self.map_origin
        message.info.origin.orientation.w = 1.0
        message.data = self.occupancy.reshape(-1).tolist()
        self.map_pub.publish(message)

    def update(self, simulation_time):
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

        stamp = self.node.get_clock().now().to_msg()
        self.camera_info.header.stamp = stamp
        self.rgb_pub.publish(self._image_message(rgb, stamp, "rgb8"))
        self.depth_pub.publish(self._image_message(depth, stamp, "32FC1"))
        self.info_pub.publish(self.camera_info)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self._estimate_visual_motion(gray, depth)
        self._integrate_depth(depth)
        self._publish_pose(stamp)
        if simulation_time - self.last_map_publish_time >= self.map_publish_interval:
            self.last_map_publish_time = float(simulation_time)
            self._publish_map(stamp)
        status = self._String()
        status.data = (
            f"{self.tracking_mode}; features={self.feature_count}; "
            f"mapped={(self.occupancy >= 0).sum()} cells"
        )
        self.status_pub.publish(status)
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def reset(self):
        self.occupancy.fill(-1)
        self.previous_gray = None
        self.previous_depth = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.previous_ground_truth = self._ground_truth_pose()
        self.map_to_base = np.eye(4, dtype=np.float64)
        self.map_to_base[2, 3] = float(self.data.xpos[self.base_id][2])
        self.map_to_camera = self.map_to_base @ self.base_to_camera
        self.path.poses.clear()
        self.tracking_mode = "INITIALIZING"
        self.feature_count = 0
        self.last_publish_time = -np.inf
        self.last_map_publish_time = -np.inf

    def close(self):
        self.renderer.close()
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
