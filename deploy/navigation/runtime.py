"""Flat indoor navigation from observed grids, verified poses, and range scans.

No camera renderer, simulator state, SDK, ROS, Torch or wall-clock singleton.
The supplied clock and packet clock_id must refer to one time domain.
"""
import math
from types import SimpleNamespace
import numpy as np
from .contracts import VelocityRequest
from .grid_navigation import GridGoalNavigator
from .motion_gateway import MotionGateway
from .terrain_navigator import TerrainNavigator
from .terminal_control import terminal_command


class PlanarSafety:
    def __init__(self, half_length=.39, half_width=.20):
        self.guard = TerrainNavigator({'body_half_length': half_length,
                                       'corridor_half_width': half_width})

    def __call__(self, command, scan):
        angles, ranges = np.asarray(scan.angles), np.asarray(scan.ranges)
        if (scan.frame_id != 'base_link' or angles.ndim != 1 or len(angles) < 24
                or ranges.shape != angles.shape or not np.isfinite(angles).all()
                or not np.isfinite(ranges).all() or not math.isfinite(scan.max_range)
                or scan.max_range < 1.0 or np.any(ranges <= 0) or np.any(ranges > scan.max_range)):
            return False
        # Full surrounding coverage is required for lateral/reverse/turn sweeps.
        ordered = np.sort(angles % (2*math.pi))
        gaps = np.diff(np.r_[ordered, ordered[0]+2*math.pi])
        if gaps.min() <= 1e-6 or gaps.max() > math.radians(16):
            return False
        lidar = SimpleNamespace(planar_angles=angles, planar_ranges=ranges,
                                planar_max_range=scan.max_range)
        return self.guard._swept_motion_clear(command, lidar)


class NavigationRuntime:
    def __init__(self, sink, clock, config=None, clock_id='monotonic'):
        config = dict(config or {})
        config['map_source'] = 'rtabmap'
        config['exact_goal'] = True
        config.setdefault('robot_radius', .45)
        config.setdefault('goal_tolerance', .15)
        self.planner = GridGoalNavigator(config)
        self.clock = clock
        self.gateway = MotionGateway(sink, clock, PlanarSafety(),
                                     limits=config.get('command_limits', (.10, .10, .20)),
                                     clock_id=clock_id)
        self.map_id, self.goal_yaw, self.terminal_active = None, None, False

    def update_map(self, map_id, grid):
        try:
            if not map_id or not isinstance(map_id, str):
                raise ValueError('A stable map session ID is required')
            values = np.asarray(grid['data'])
            if (values.shape != (grid['height'], grid['width']) or values.size == 0
                    or not np.isfinite(values).all() or np.any(values < -1)
                    or np.any(values > 100) or np.any(values != np.floor(values))
                    or not np.isfinite(grid['origin']).all() or len(grid['origin']) != 2
                    or not math.isfinite(grid['resolution']) or grid['resolution'] <= 0):
                raise ValueError('Invalid occupancy grid')
            if map_id != self.map_id:
                self.cancel('map_session_changed')
            if not self.planner.update_occupancy_grid(grid):
                raise ValueError('No observed free map cells')
            self.map_id = map_id
            # A new map can invalidate the previous path before the replan timer.
            self.planner.last_plan_time = -math.inf
        except Exception:
            self.cancel('invalid_map')
            self.map_id = None
            self.planner.map_ready = False
            raise

    def set_goal(self, goal, frame, yaw=None):
        if (len(goal) != 2 or not all(math.isfinite(x) for x in goal)
                or (yaw is not None and not math.isfinite(yaw))):
            self.cancel('invalid_goal')
            raise ValueError('Invalid goal pose')
        if (not self.planner.map_ready or frame.pose is None or frame.pose.map_id != self.map_id
                or not self.gateway._pose_ready(frame, self.clock())):
            self.cancel('verified_map_required')
            raise ValueError('Goal requires a fresh pose in the loaded map')
        self.gateway.stop('new_goal_requires_arm')
        self.goal_yaw, self.terminal_active = yaw, False
        return self.planner.set_goal(goal, frame.pose.xy_yaw[:2], self.clock())

    def arm(self, frame):
        if not self.planner.active or frame.pose is None or frame.pose.map_id != self.map_id:
            raise ValueError('Active goal and matching map required')
        return self.gateway.arm(frame)

    def cancel(self, reason='cancelled'):
        self.planner.cancel()
        self.terminal_active = False
        return self.gateway.stop(reason)

    def step(self, frame, manual=None):
        try:
            return self._step(frame, manual)
        except Exception:
            return self.gateway.stop('invalid_navigation_input')

    def _step(self, frame, manual=None):
        if manual is not None:
            self.planner.cancel()  # Release never silently resumes the old route.
            self.terminal_active = False
            return self.gateway.dispatch(frame, manual=manual)
        health = self.gateway.check_health(frame)
        if health is not None:
            return health
        issued = self.clock()
        epoch = self.gateway.epoch
        if not self.gateway.armed:
            return self.gateway.dispatch(frame)
        if (frame.pose is None or frame.pose.map_id != self.map_id or not self.planner.map_ready
                or not self.gateway._pose_ready(frame, issued)):
            return self.gateway.stop('localization_unavailable')
        if not self.planner._world_in_bounds(frame.pose.xy_yaw[:2]):
            return self.gateway.stop('pose_outside_observed_map')
        if not self.planner.active:
            return self.gateway.stop('goal_reached' if self.planner.reached else 'no_goal')
        try:
            if len(frame.velocity_body) != 3 or not all(math.isfinite(v) for v in frame.velocity_body):
                raise ValueError('Invalid measured velocity')
            pose = np.asarray(frame.pose.xy_yaw)
            command, self.terminal_active = terminal_command(
                pose, self.planner.goal, self.goal_yaw, frame.velocity_body,
                self.terminal_active, self.planner.goal_tolerance)
            if command is not None and np.linalg.norm(command[:2]) > 0:
                current = self.planner._world_to_grid(pose[:2])
                target = self.planner._world_to_grid(self.planner.goal)
                if not self.planner._line_clear(current, target):
                    return self.gateway.stop('terminal_path_not_observed_free')
            if command is None:
                command = self.planner.update(pose[:2], pose[2], issued)
            if self.planner.reached:
                return self.gateway.stop('goal_reached')
            request = VelocityRequest(issued, tuple(float(v) for v in command), epoch)
        except Exception:
            return self.gateway.stop('planner_error')
        return self.gateway.dispatch(frame, autonomous=request)
