"""A* goal navigation over static MuJoCo or optimized RTAB-Map grids."""

import heapq
import math

import cv2
import mujoco
import numpy as np


def _angle_difference(angle, reference):
    return math.atan2(math.sin(angle - reference), math.cos(angle - reference))


class GoalNavigator:
    """Plan collision-inflated global paths and follow them with velocity commands."""

    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.enabled = bool(config.get("enabled", True))
        self.resolution = float(config.get("resolution", 0.15))
        self.robot_radius = float(config.get("robot_radius", 0.24))
        self.max_step_height = float(config.get("max_step_height", 0.18))
        self.max_speed = float(config.get("max_speed", 0.75))
        self.max_yaw_rate = float(config.get("max_yaw_rate", 0.9))
        self.goal_tolerance = float(config.get("goal_tolerance", 0.30))
        self.waypoint_tolerance = float(config.get("waypoint_tolerance", 0.32))
        self.turn_in_place_angle = math.radians(
            float(config.get("turn_in_place_angle_deg", 38.0))
        )
        self.replan_interval = float(config.get("replan_interval", 3.0))
        self.preferred_clearance = float(config.get("preferred_clearance", 0.38))
        self.clearance_cost_weight = float(
            config.get("clearance_cost_weight", 0.45)
        )
        self.smoothing_clearance = float(
            config.get("smoothing_clearance", 0.30)
        )
        self.terrain_group = int(config.get("terrain_group", 1))
        self.map_source = str(config.get("map_source", "mujoco")).lower()
        self.dynamic_map = self.map_source in ("rtabmap", "vslam", "ros")
        self.occupied_threshold = int(config.get("occupied_threshold", 50))
        self.bounds = np.asarray(
            config.get("bounds", [-2.75, 10.75, -5.75, 5.75]),
            dtype=np.float64,
        )
        self.presets = {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in config.get("presets", {}).items()
        }

        if self.dynamic_map:
            self.x_values = np.empty(0, dtype=np.float64)
            self.y_values = np.empty(0, dtype=np.float64)
            self.raw_occupied = np.empty((0, 0), dtype=bool)
            self.known = np.empty((0, 0), dtype=bool)
            self.clearance = np.empty((0, 0), dtype=np.float32)
            self.occupied = np.empty((0, 0), dtype=bool)
            self.map_ready = False
        else:
            self.x_values = np.arange(
                self.bounds[0], self.bounds[1] + 1.0e-9, self.resolution
            )
            self.y_values = np.arange(
                self.bounds[2], self.bounds[3] + 1.0e-9, self.resolution
            )
            self.raw_occupied, self.clearance, self.occupied = (
                self._build_occupancy_grid()
            )
            self.known = np.ones_like(self.raw_occupied, dtype=bool)
            self.map_ready = True
        self.map_revision = 0

        self.active = False
        self.reached = False
        self.failed = False
        self.goal = np.zeros(2, dtype=np.float64)
        self.effective_goal = np.zeros(2, dtype=np.float64)
        self.goal_name = ""
        self.path = []
        self.waypoint_index = 0
        self.last_plan_time = -math.inf
        self.last_distance = math.inf

    def update_occupancy_grid(self, grid):
        """Replace the planning grid with RTAB-Map's optimized global map."""
        if not self.dynamic_map or grid is None:
            return False
        width = int(grid["width"])
        height = int(grid["height"])
        resolution = float(grid["resolution"])
        data = np.asarray(grid["data"], dtype=np.int8)
        if (
            width <= 0
            or height <= 0
            or resolution <= 0.0
            or data.shape != (height, width)
        ):
            return False

        origin_x, origin_y = grid["origin"]
        self.resolution = resolution
        self.x_values = origin_x + (np.arange(width) + 0.5) * resolution
        self.y_values = origin_y + (np.arange(height) + 0.5) * resolution
        self.bounds = np.array(
            [
                self.x_values[0],
                self.x_values[-1],
                self.y_values[0],
                self.y_values[-1],
            ],
            dtype=np.float64,
        )

        known = data >= 0
        # Unknown cells are not navigable. They become free only after the
        # RGB-D mapper has actually observed them.
        raw_yx = (~known) | (data >= self.occupied_threshold)
        self.raw_occupied = raw_yx.T.copy()
        self.known = known.T.copy()
        free_yx = (~raw_yx).astype(np.uint8)
        self.clearance = (
            cv2.distanceTransform(
                free_yx, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ).T
            * resolution
        ).astype(np.float32)
        self.occupied = self.raw_occupied | (self.clearance < self.robot_radius)
        self.map_ready = bool(np.any(known & (data < self.occupied_threshold)))
        self.map_revision += 1
        return self.map_ready

    @property
    def preset_names(self):
        return list(self.presets.keys())

    def _build_occupancy_grid(self):
        raw = np.zeros((len(self.x_values), len(self.y_values)), dtype=np.uint8)
        geomgroup = np.zeros(6, dtype=np.uint8)
        geomgroup[self.terrain_group] = 1
        geomid = np.array([-1], dtype=np.int32)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ray_height = 3.0

        for ix, x in enumerate(self.x_values):
            for iy, y in enumerate(self.y_values):
                origin = np.array([x, y, ray_height], dtype=np.float64)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    geomgroup,
                    True,
                    -1,
                    geomid,
                )
                if distance < 0.0:
                    raw[ix, iy] = 1
                    continue
                hit_height = ray_height - distance
                hit_geom = int(geomid[0])
                if hit_geom >= 0 and self.model.geom_contype[hit_geom] == 0:
                    continue
                if hit_height > self.max_step_height:
                    raw[ix, iy] = 1

        # Threshold a Euclidean clearance field instead of rounding the radius
        # up to a whole dilation kernel. At 0.15 m/cell, the old 0.24 m radius
        # became 0.30 m and closed visually passable indoor gaps.
        free = (raw.T == 0).astype(np.uint8)
        clearance = (
            cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).T
            * self.resolution
        )
        inflated = raw.astype(bool) | (clearance < self.robot_radius)
        return raw.astype(bool), clearance.astype(np.float32), inflated

    def _world_to_grid(self, point):
        ix = int(round((float(point[0]) - self.bounds[0]) / self.resolution))
        iy = int(round((float(point[1]) - self.bounds[2]) / self.resolution))
        return (
            int(np.clip(ix, 0, len(self.x_values) - 1)),
            int(np.clip(iy, 0, len(self.y_values) - 1)),
        )

    def _world_in_bounds(self, point):
        return bool(
            self.map_ready
            and self.bounds[0] <= float(point[0]) <= self.bounds[1]
            and self.bounds[2] <= float(point[1]) <= self.bounds[3]
        )

    def _grid_to_world(self, cell):
        return np.array(
            [self.x_values[cell[0]], self.y_values[cell[1]]], dtype=np.float64
        )

    def _nearest_free(self, cell):
        if not self.occupied[cell]:
            return cell
        max_radius = max(self.occupied.shape)
        for radius in range(1, max_radius):
            candidates = []
            for dx in range(-radius, radius + 1):
                candidates.append((cell[0] + dx, cell[1] - radius))
                candidates.append((cell[0] + dx, cell[1] + radius))
            for dy in range(-radius + 1, radius):
                candidates.append((cell[0] - radius, cell[1] + dy))
                candidates.append((cell[0] + radius, cell[1] + dy))
            for candidate in candidates:
                if (
                    0 <= candidate[0] < self.occupied.shape[0]
                    and 0 <= candidate[1] < self.occupied.shape[1]
                    and not self.occupied[candidate]
                ):
                    return candidate
        return None

    @staticmethod
    def _heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _astar(self, start, goal):
        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        queue = [(self._heuristic(start, goal), 0.0, start)]
        came_from = {}
        cost = {start: 0.0}

        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            if current_cost > cost.get(current, math.inf):
                continue
            for dx, dy, move_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not (
                    0 <= nxt[0] < self.occupied.shape[0]
                    and 0 <= nxt[1] < self.occupied.shape[1]
                ):
                    continue
                if self.occupied[nxt]:
                    continue
                if dx and dy:
                    if self.occupied[current[0] + dx, current[1]] or self.occupied[
                        current[0], current[1] + dy
                    ]:
                        continue
                clearance_deficit = max(
                    0.0, self.preferred_clearance - float(self.clearance[nxt])
                )
                clearance_penalty = self.clearance_cost_weight * (
                    clearance_deficit / self.resolution
                ) ** 2
                new_cost = current_cost + move_cost + clearance_penalty
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    came_from[nxt] = current
                    priority = new_cost + self._heuristic(nxt, goal)
                    heapq.heappush(queue, (priority, new_cost, nxt))
        return []

    def _line_clear(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy), 1)
        for step in range(steps + 1):
            alpha = step / steps
            ix = int(round(start[0] + alpha * dx))
            iy = int(round(start[1] + alpha * dy))
            if (
                self.occupied[ix, iy]
                or self.clearance[ix, iy] < self.smoothing_clearance
            ):
                return False
        return True

    def _simplify(self, cells):
        if len(cells) <= 2:
            return cells
        simplified = [cells[0]]
        index = 0
        while index < len(cells) - 1:
            candidate = len(cells) - 1
            while candidate > index + 1 and not self._line_clear(
                cells[index], cells[candidate]
            ):
                candidate -= 1
            simplified.append(cells[candidate])
            index = candidate
        return simplified

    def plan(self, start_position, simulation_time=0.0):
        if not self.map_ready:
            self.failed = True
            self.active = False
            self.path = []
            return False
        if self.dynamic_map and (
            not self._world_in_bounds(start_position)
            or not self._world_in_bounds(self.goal)
        ):
            self.failed = True
            self.active = False
            self.path = []
            return False
        start_cell = self._world_to_grid(start_position)
        goal_cell = self._world_to_grid(self.goal)
        if self.dynamic_map and not self.known[goal_cell]:
            self.failed = True
            self.active = False
            self.path = []
            return False
        # A forward-facing RGB-D camera does not observe the footprint under
        # (or immediately behind) the robot, so the exact current map cell can
        # legitimately remain unknown.  `_nearest_free()` snaps only the
        # start to the closest observed, inflated-free cell; unknown cells are
        # still never traversed and the requested goal must itself be known.
        start = self._nearest_free(start_cell)
        goal = self._nearest_free(goal_cell)
        if start is None or goal is None:
            self.failed = True
            self.active = False
            self.path = []
            return False
        cells = self._astar(start, goal)
        if not cells:
            self.failed = True
            self.active = False
            self.path = []
            return False
        cells = self._simplify(cells)
        self.path = [self._grid_to_world(cell) for cell in cells]
        # A requested point can lie inside a sofa/table footprint.  Navigate
        # to the nearest inflated free cell instead of putting the exact,
        # unreachable coordinate back into the simplified path.
        self.effective_goal[:] = self.path[-1]
        self.waypoint_index = 1 if len(self.path) > 1 else 0
        self.last_plan_time = float(simulation_time)
        self.failed = False
        return True

    def set_goal(self, goal, start_position, simulation_time=0.0, name="coordinate"):
        if not self.enabled:
            return False
        self.goal[:] = np.asarray(goal, dtype=np.float64)[:2]
        self.goal_name = str(name)
        self.active = True
        self.reached = False
        self.failed = False
        success = self.plan(start_position, simulation_time)
        if success:
            print(
                f"Navigation goal '{self.goal_name}': "
                f"x={self.goal[0]:.2f}, y={self.goal[1]:.2f}, "
                f"waypoints={len(self.path)}"
            )
        else:
            print(f"Navigation goal '{self.goal_name}' has no free path")
        return success

    def set_named_goal(self, name, start_position, simulation_time=0.0):
        if name not in self.presets:
            return False
        return self.set_goal(
            self.presets[name], start_position, simulation_time, name=name
        )

    def cancel(self):
        self.active = False
        self.reached = False
        self.failed = False
        self.path = []
        self.waypoint_index = 0

    def update(self, position, yaw, simulation_time):
        if not self.active:
            return np.zeros(3, dtype=np.float32)
        position = np.asarray(position, dtype=np.float64)[:2]
        to_goal = self.effective_goal - position
        distance = float(np.linalg.norm(to_goal))
        self.last_distance = distance
        if distance <= self.goal_tolerance:
            self.active = False
            self.reached = True
            print(f"Navigation goal '{self.goal_name}' reached")
            return np.zeros(3, dtype=np.float32)

        if float(simulation_time) - self.last_plan_time >= self.replan_interval:
            # The map is static.  Keep the current side of a passable corridor
            # instead of rebuilding an equally good path on the opposite side
            # every few seconds, which presents as left-right wandering.
            waypoint = self.path[min(self.waypoint_index, len(self.path) - 1)]
            current_cell = self._world_to_grid(position)
            waypoint_cell = self._world_to_grid(waypoint)
            if self._line_clear(current_cell, waypoint_cell):
                self.last_plan_time = float(simulation_time)
            else:
                self.plan(position, simulation_time)
                if not self.active:
                    return np.zeros(3, dtype=np.float32)

        while self.waypoint_index < len(self.path) - 1:
            waypoint_distance = np.linalg.norm(
                self.path[self.waypoint_index] - position
            )
            if waypoint_distance > self.waypoint_tolerance:
                break
            self.waypoint_index += 1

        target = self.path[self.waypoint_index]
        delta = target - position
        desired_yaw = math.atan2(delta[1], delta[0])
        heading_error = _angle_difference(desired_yaw, float(yaw))
        yaw_rate = float(
            np.clip(1.8 * heading_error, -self.max_yaw_rate, self.max_yaw_rate)
        )
        if abs(heading_error) > self.turn_in_place_angle:
            forward = 0.0
        else:
            forward = min(self.max_speed, max(0.22, 0.8 * distance))
            forward *= max(0.30, math.cos(heading_error))
        return np.array([forward, 0.0, yaw_rate], dtype=np.float32)

    def append_path(self, scene):
        if scene is None or not self.path:
            return
        start = min(self.waypoint_index, len(self.path) - 1)
        for index, point in enumerate(self.path[start:]):
            if scene.ngeom >= scene.maxgeom:
                break
            is_goal = index == len(self.path[start:]) - 1
            rgba = np.array(
                [1.0, 0.25, 0.05, 0.95] if is_goal else [0.10, 1.0, 0.20, 0.85],
                dtype=np.float32,
            )
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.045 if is_goal else 0.025, 0.0, 0.0]),
                pos=np.array([point[0], point[1], 0.06]),
                mat=np.eye(3).ravel(),
                rgba=rgba,
            )
            scene.ngeom += 1

    def status_text(self):
        if self.dynamic_map and not self.map_ready:
            return "GOAL: waiting for RTAB-Map occupancy grid"
        if self.active:
            return (
                f"GOAL: {self.goal_name} ({self.goal[0]:.1f}, {self.goal[1]:.1f})  "
                f"distance={self.last_distance:.2f}m  "
                f"waypoint={self.waypoint_index + 1}/{len(self.path)}"
            )
        if self.reached:
            return f"GOAL: {self.goal_name} REACHED"
        if self.failed:
            return f"GOAL: {self.goal_name} NO PATH"
        return "GOAL: manual control"
