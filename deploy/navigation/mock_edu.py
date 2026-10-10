"""Deterministic, in-process EDU fixture. No network or physical dynamics."""
import math
import numpy as np
from .contracts import NavigationFrame, PoseEstimate, RangeScan, RobotState


class FakeClock:
    def __init__(self):
        self.time = 0.

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


class MockEduClient:
    """SDK-shaped transport: accepts commands; simulated plant advances separately."""
    def __init__(self, clock):
        self.clock = clock
        self.velocity = (0., 0., 0.)
        self.moves, self.stops = [], 0
        self.move_code, self.stop_code, self.delay = 0, 0, 0.
        self.raise_move = False

    def Move(self, *velocity):
        self.moves.append((self.clock(), tuple(velocity)))
        # Model an ambiguous failure: the robot may receive a failed RPC.
        self.velocity = tuple(velocity)
        self.clock.advance(self.delay)
        if self.raise_move:
            raise RuntimeError('Injected transport exception after accepting command')
        return self.move_code

    def StopMove(self):
        self.stops += 1
        if self.stop_code == 0:
            self.velocity = (0., 0., 0.)
        return self.stop_code


class FlatWorld:
    """2D kinematics and synthetic grid rays; explicitly not a SLAM/gait test."""
    def __init__(self, clock, client, yaw=0.):
        self.clock, self.client = clock, client
        self.pose = np.array([0., 0., yaw])
        self.resolution, self.origin = .05, np.array([-1., -2.5])
        self.cells = np.zeros((100, 120), dtype=np.int8)
        self.cells[[0, -1], :] = 100
        self.cells[:, [0, -1]] = 100
        # A wall with a 1.8 m doorway, comfortably above the configured footprint.
        self.cells[:, 50:52] = 100
        self.cells[32:68, 50:52] = 0
        self.angles = np.linspace(-math.pi, math.pi, 120, endpoint=False)
        self.sequence = 0
        self.collisions = 0

    def grid(self):
        return dict(data=self.cells.copy(), width=120, height=100,
                    origin=tuple(self.origin), resolution=self.resolution)

    def ranges(self):
        distances = np.arange(.025, 4.001, .025)
        angles = self.angles + self.pose[2]
        points = self.pose[:2] + distances[:, None, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1)[None, :, :]
        cells = np.floor((points - self.origin) / self.resolution).astype(int)
        outside = ((cells[..., 0] < 0) | (cells[..., 0] >= 120)
                   | (cells[..., 1] < 0) | (cells[..., 1] >= 100))
        occupied = outside | (self.cells[np.clip(cells[..., 1], 0, 99),
                                        np.clip(cells[..., 0], 0, 119)] != 0)
        first = occupied.argmax(axis=0)
        return np.where(occupied.any(axis=0), distances[first], 4.)

    def frame(self):
        self.sequence += 1
        t = self.clock()
        return NavigationFrame(RobotState(t, self.sequence, True, True),
                               PoseEstimate(t, tuple(self.pose), 'mock-map', True),
                               RangeScan(t, tuple(self.angles), tuple(self.ranges()), 4.),
                               self.client.velocity)

    def advance(self, seconds=.05):
        vx, vy, wz = self.client.velocity
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose += np.array([c*vx-s*vy, s*vx+c*vy, wz]) * seconds
        self.clock.advance(seconds)
        # Independent occupied-cell/rectangle overlap check, conservative by half a cell.
        ys, xs = np.where(self.cells != 0)
        points = self.origin + self.resolution*np.column_stack((xs+.5, ys+.5))
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        local = (points-self.pose[:2]) @ np.array([[c, -s], [s, c]])
        if np.any((abs(local[:, 0]) < .39+.025) & (abs(local[:, 1]) < .20+.025)):
            self.collisions += 1
