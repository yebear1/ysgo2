"""Opt-in MuJoCo entry adapter for the same low-speed runtime as mock EDU."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.adapters import MujocoSensorAdapter, MujocoVelocitySink
from navigation.contracts import VelocityRequest
from navigation.runtime import NavigationRuntime


class SharedNavigationSession:
    def __init__(self, clock, config=None, map_id='simulation-map'):
        self.clock, self.map_id = clock, map_id
        self.sink = MujocoVelocitySink()
        self.sensors = MujocoSensorAdapter(map_id)
        self.runtime = NavigationRuntime(self.sink, clock, config, clock_id='simulation')
        self.pending_goal = None

    def update_map(self, grid):
        try:
            self.runtime.update_map(self.map_id, grid)
        except (ValueError, TypeError, KeyError):
            self.pending_goal = None

    def request_goal(self, xy):
        self.runtime.cancel('new_goal_requested')
        self.pending_goal = tuple(xy)

    def step(self, bridge, lidar, manual=None, cancel=False):
        try:
            frame = self.sensors.read(self.clock(), bridge, lidar, enabled=True)
        except Exception:
            self.pending_goal = None
            return self.runtime.cancel('sensor_adapter_error')
        if cancel or manual is not None:
            self.pending_goal = None
            self.runtime.cancel('operator_takeover')
        if self.pending_goal is not None and self.runtime.planner.map_ready:
            # Initial localization can wait; once an armed run stops, no retry.
            if self.runtime.gateway._pose_ready(frame, self.clock()):
                goal, self.pending_goal = self.pending_goal, None
                try:
                    if self.runtime.set_goal(goal, frame):
                        self.runtime.arm(frame)
                except ValueError:
                    self.runtime.cancel('goal_not_ready')
        request = None if manual is None else VelocityRequest(self.clock(), tuple(manual))
        return self.runtime.step(frame, request)
