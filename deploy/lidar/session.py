"""Shared live/replay analysis session; output stays in memory, never DDS.

This deliberately models an enabled operator only for command analysis. A
future real transport must supply independently verified RobotState and an
explicit enable; sensor arrival itself never authorizes real motion.
"""
from dataclasses import asdict
import math
import time
from .localization import GridLocalizer
from ..navigation.contracts import NavigationFrame, RangeScan, RobotState
from ..navigation.runtime import NavigationRuntime


class AnalysisSink:
    def __init__(self):
        self.command_count = 0
        self.last = (0., 0., 0.)

    def send(self, velocity):
        self.command_count += 1
        self.last = tuple(velocity)
        return True

    def stop(self):
        self.last = (0., 0., 0.)
        return True


class AnalysisSession:
    def __init__(self, saved_map, search_center, search_radius=1., goal=None):
        self.map = saved_map
        self.goal = goal
        self.now = 0.
        self.last_event_time = -math.inf
        self.sequence = 0
        self.ever_armed = False
        self.sink = AnalysisSink()
        self.runtime = NavigationRuntime(self.sink, lambda: self.now, clock_id='replay')
        self.runtime.update_map(saved_map.map_id, saved_map.grid)
        self.localizer = GridLocalizer(saved_map, search_center, search_radius)

    def process(self, event):
        """Preserve sensor age, account for CPU work, never rearm after a stop."""
        t = event['t']
        if type(t) not in (int, float) or not math.isfinite(t) or t < self.last_event_time:
            self.runtime.cancel('recording_time_invalid')
            raise ValueError('Recording time must be finite and monotonic')
        self.last_event_time = t
        self.now = max(self.now, t)
        if event.get('type') == 'fault':
            self.localizer.reset(event.get('reason', 'sensor_fault'))
            decision = self.runtime.cancel('sensor_fault')
            return dict(t=t, reason=self.localizer.last_result.reason, decision=asdict(decision))
        if event.get('type') != 'frame':
            raise ValueError('Unknown event type')
        scan = RangeScan(float(event['scan_stamp']), tuple(event['angles']),
                         tuple(event['ranges']), float(event['max_range']))
        odom_stamp = float(event['odom_stamp'])
        before = time.perf_counter()
        if (not math.isfinite(odom_stamp) or not 0 <= self.now-odom_stamp <= .20
                or abs(odom_stamp-scan.stamp) > .05):
            match = self.localizer.reset('odometry_stale_or_unsynchronized')
        else:
            match = self.localizer.update(scan, event['odom_xy_yaw'], self.now)
        elapsed = time.perf_counter()-before
        self.now += elapsed
        self.sequence += 1
        frame = NavigationFrame(RobotState(t, self.sequence, True, self.goal is not None),
                                match.pose, scan, tuple(event.get('velocity_body', (0., 0., 0.))), 'replay')
        if (self.goal is not None and not self.ever_armed and match.pose is not None
                and match.pose.matched and self.runtime.gateway._pose_ready(frame, self.now)):
            try:
                self.runtime.set_goal(self.goal, frame)
                self.runtime.arm(frame)
                self.ever_armed = True
            except ValueError:
                pass
        decision = self.runtime.step(frame)
        return dict(t=t, reason=match.reason, score=match.score, inlier_ratio=match.inlier_ratio,
                    ambiguity_gap=match.ambiguity_gap, confirmations=match.confirmations,
                    pose=None if match.pose is None else asdict(match.pose),
                    processing_seconds=elapsed, decision=asdict(decision))

    def close(self):
        return self.runtime.cancel('end_of_recording')
