"""MuJoCo map construction and drawing adapter for the shared planner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import mujoco
import numpy as np
from navigation.grid_navigation import GridGoalNavigator, _angle_difference


class GoalNavigator(GridGoalNavigator):
    def __init__(self, model, data, config):
        self.model, self.data = model, data
        config = dict(config)
        config.setdefault("map_source", "mujoco")
        super().__init__(config)

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
