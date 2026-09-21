import cv2
import mujoco
import numpy as np


class LidarHeightMap:
    """Body-centric multi-beam ray scanner and elevation-grid builder."""

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.base_id = model.body(config.get("body_name", "base")).id
        self.sensor_offset = np.asarray(
            config.get("sensor_offset", [0.20, 0.0, 0.12]), dtype=np.float64
        )
        self.x_values = np.arange(
            config.get("x_min", 0.25),
            config.get("x_max", 2.05) + 1e-9,
            config.get("resolution", 0.10),
        )
        self.y_values = np.arange(
            config.get("y_min", -0.75),
            config.get("y_max", 0.75) + 1e-9,
            config.get("resolution", 0.10),
        )
        self.scan_interval = float(config.get("scan_interval", 0.10))
        self.rise_threshold = float(config.get("rise_threshold", 0.045))
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        self.geomgroup[int(config.get("terrain_group", 1))] = 1

        shape = (len(self.x_values), len(self.y_values))
        self.heights = np.full(shape, np.nan, dtype=np.float32)
        self.points = np.full((*shape, 3), np.nan, dtype=np.float64)
        self.elevation = np.full(shape, np.nan, dtype=np.float32)
        self.baseline = 0.0
        self.max_rise = 0.0
        self.nearest_rise = np.inf
        self.stairs_detected = False
        self.image = np.zeros((180, 240, 3), dtype=np.uint8)
        self.image_dirty = False
        self._last_base_pos = np.zeros(3, dtype=np.float64)
        self._last_yaw = 0.0
        self.planar_max_range = float(config.get("planar_max_range", 2.0))
        self.planar_angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        self.planar_ranges = np.full(
            self.planar_angles.shape, self.planar_max_range, dtype=np.float32
        )

        # The CTS teacher was trained with this exact 17 x 11 body-yaw grid.
        # Keep it separate from the longer-range display grid above: this grid
        # is the actual exteroceptive input used to choose joint actions.
        self.policy_x_values = np.asarray(
            config.get("policy_points_x", np.arange(-0.8, 0.81, 0.1)),
            dtype=np.float64,
        )
        self.policy_y_values = np.asarray(
            config.get("policy_points_y", np.arange(-0.5, 0.51, 0.1)),
            dtype=np.float64,
        )
        self.policy_heights = np.zeros(
            (len(self.policy_x_values), len(self.policy_y_values)), dtype=np.float32
        )

    def scan(self):
        base_pos = self.data.xpos[self.base_id].copy()
        rotation = self.data.xmat[self.base_id].reshape(3, 3).copy()
        origin = base_pos + rotation @ self.sensor_offset
        geomid = np.array([-1], dtype=np.int32)

        ground_distance = mujoco.mj_ray(
            self.model,
            self.data,
            origin,
            np.array([0.0, 0.0, -1.0]),
            self.geomgroup,
            True,
            -1,
            geomid,
        )
        if ground_distance >= 0.0:
            self.baseline = float(origin[2] - ground_distance)
        else:
            self.baseline = float(base_pos[2] - 0.35)

        self.heights.fill(np.nan)
        self.points.fill(np.nan)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self._last_base_pos[:] = base_pos
        self._last_yaw = float(yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        # The elevation map is reconstructed from the simulated 3-D point
        # cloud as vertical map cells. This makes high obstacles and negative
        # drop-offs measurable without assuming a specific terrain type.
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for i, x in enumerate(self.x_values):
            for j, y in enumerate(self.y_values):
                target_world = base_pos + yaw_rotation @ np.array([x, y, 0.0])
                ray_origin = np.array(
                    [target_world[0], target_world[1], base_pos[2] + 2.0]
                )
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    ray_origin,
                    direction,
                    self.geomgroup,
                    True,
                    -1,
                    geomid,
                )
                if distance >= 0.0:
                    hit = ray_origin + distance * direction
                    self.points[i, j] = hit
                    self.heights[i, j] = hit[2]

        self.elevation[:] = self.heights - self.baseline

        corridor = np.abs(self.y_values) <= 0.40
        ahead = self.elevation[:, corridor]
        finite_ahead = ahead[np.isfinite(ahead)]
        self.max_rise = float(np.max(finite_ahead)) if finite_ahead.size else 0.0

        self.nearest_rise = np.inf
        for i, x in enumerate(self.x_values):
            row = ahead[i]
            row = row[np.isfinite(row)]
            if row.size and np.percentile(row, 70) > self.rise_threshold:
                self.nearest_rise = float(x)
                break
        self.stairs_detected = bool(
            self.max_rise > self.rise_threshold and np.isfinite(self.nearest_rise)
        )
        self._scan_policy_grid(base_pos, rotation)
        self._scan_planar_ring(base_pos, yaw)
        self.image = self._make_image()
        self.image_dirty = True

    def _scan_planar_ring(self, base_pos, yaw):
        """Measure walls and furniture around the complete turning envelope."""
        geomid = np.array([-1], dtype=np.int32)
        origin = np.asarray(base_pos, dtype=np.float64).copy()
        origin[2] = max(self.baseline + 0.32, 0.32)
        for index, angle in enumerate(self.planar_angles):
            world_angle = yaw + float(angle)
            direction = np.array(
                [np.cos(world_angle), np.sin(world_angle), 0.0], dtype=np.float64
            )
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                self.geomgroup,
                True,
                -1,
                geomid,
            )
            self.planar_ranges[index] = (
                min(float(distance), self.planar_max_range)
                if distance >= 0.0
                else self.planar_max_range
            )

    def planar_clearance(self, center_angle=0.0, half_angle=np.pi):
        difference = np.arctan2(
            np.sin(self.planar_angles - float(center_angle)),
            np.cos(self.planar_angles - float(center_angle)),
        )
        selected = np.abs(difference) <= float(half_angle)
        if not np.any(selected):
            return self.planar_max_range
        return float(np.min(self.planar_ranges[selected]))

    def translation_clearance(
        self,
        center_angle,
        half_width=0.23,
        body_half_length=0.39,
    ):
        """Free travel beyond a rectangular footprint in one direction."""
        relative = np.arctan2(
            np.sin(self.planar_angles - float(center_angle)),
            np.cos(self.planar_angles - float(center_angle)),
        )
        longitudinal = self.planar_ranges * np.cos(relative)
        lateral = self.planar_ranges * np.sin(relative)
        selected = (longitudinal > 0.0) & (np.abs(lateral) <= half_width)
        if not np.any(selected):
            return self.planar_max_range - body_half_length
        return float(np.min(longitudinal[selected] - body_half_length))

    def _scan_policy_grid(self, base_pos, rotation):
        """Build the 187-point terrain observation used during RL training."""
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        geomid = np.array([-1], dtype=np.int32)
        ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        for i, x in enumerate(self.policy_x_values):
            for j, y in enumerate(self.policy_y_values):
                xy_world = base_pos + yaw_rotation @ np.array([x, y, 0.0])
                origin = np.array([xy_world[0], xy_world[1], base_pos[2] + 1.0])
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    ray_direction,
                    self.geomgroup,
                    True,
                    -1,
                    geomid,
                )
                self.policy_heights[i, j] = (
                    origin[2] - distance if distance >= 0.0 else self.baseline
                )

    def policy_height_observation(self, base_height):
        """Return height features with the same order/scaling as Go2 training."""
        heights = self.policy_heights.reshape(-1)
        return np.clip(base_height - 0.5 - heights, -1.0, 1.0).astype(
            np.float32
        ) * 2.5

    def sample_corridor(self, angle, x_values, y_values):
        """Scan a candidate walking corridor at an angle relative to the body."""
        angle_world = self._last_yaw + float(angle)
        ca, sa = np.cos(angle_world), np.sin(angle_world)
        corridor_rotation = np.array(
            [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        geomid = np.array([-1], dtype=np.int32)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        result = np.full((len(x_values), len(y_values)), np.nan, dtype=np.float32)
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                point = self._last_base_pos + corridor_rotation @ np.array([x, y, 0.0])
                origin = np.array([point[0], point[1], self._last_base_pos[2] + 2.0])
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    self.geomgroup,
                    True,
                    -1,
                    geomid,
                )
                if distance >= 0.0:
                    result[i, j] = origin[2] - distance - self.baseline
        return result

    def status_text(self):
        mode = "ELEVATED TERRAIN" if self.stairs_detected else "FLAT"
        distance = "--" if not np.isfinite(self.nearest_rise) else f"{self.nearest_rise:.2f}m"
        return (
            f"3D LiDAR: {mode}  POLICY: TERRAIN-AWARE\n"
            f"rise={self.max_rise:.2f}m  nearest={distance}\n"
            f"map={len(self.x_values)}x{len(self.y_values)}  "
            f"policy={len(self.policy_x_values)}x{len(self.policy_y_values)}"
        )

    def append_point_cloud(self, scene):
        if scene is None:
            return
        for i in range(0, len(self.x_values), 2):
            for j in range(0, len(self.y_values), 2):
                point = self.points[i, j]
                if not np.all(np.isfinite(point)) or scene.ngeom >= scene.maxgeom:
                    continue
                height = float(np.nan_to_num(self.elevation[i, j], nan=0.0))
                value = np.clip(height / 0.50, 0.0, 1.0)
                rgba = np.array([value, 1.0 - value, 0.15, 0.85], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.012, 0.0, 0.0]),
                    pos=point,
                    mat=np.eye(3).ravel(),
                    rgba=rgba,
                )
                scene.ngeom += 1

    def consume_image(self):
        if not self.image_dirty:
            return None
        self.image_dirty = False
        return self.image

    def _make_image(self):
        values = np.nan_to_num(self.elevation, nan=-0.05)
        normalized = np.clip((values + 0.05) / 0.55, 0.0, 1.0)
        gray = (normalized * 255).astype(np.uint8)
        # Forward is up, robot is centered at the bottom of the map.
        gray = np.flipud(gray)
        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color = cv2.resize(color, (240, 180), interpolation=cv2.INTER_NEAREST)
        label = "ELEVATED" if self.stairs_detected else "FLAT"
        cv2.rectangle(color, (0, 0), (239, 26), (8, 8, 8), -1)
        cv2.putText(
            color,
            f"ELEVATION GRID  {label}",
            (7, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return color
