"""Injected transport/sensor adapters; importing this never opens DDS or ROS."""
from .contracts import NavigationFrame, PoseEstimate, RangeScan, RobotState


class EduSportSink:
    def __init__(self, client):
        self.client = client

    def send(self, velocity):
        return self.client.Move(*velocity) == 0

    def stop(self):
        return self.client.StopMove() == 0


class MujocoVelocitySink:
    """Consumed by the simulation's gait loop; no direct qpos/torque writes."""
    def __init__(self):
        self.velocity = (0., 0., 0.)

    def send(self, velocity):
        self.velocity = tuple(velocity)
        return True

    def stop(self):
        self.velocity = (0., 0., 0.)
        return True


class MujocoSensorAdapter:
    """Reads VSLAM products + measured scans, with no simulator-pose fallback."""
    def __init__(self, map_id):
        self.map_id, self.sequence = map_id, 0

    def read(self, simulation_time, bridge, lidar, enabled, emergency=False):
        self.sequence += 1
        pose = None
        if bridge is not None and bridge.global_pose is not None:
            stamp = bridge.global_pose_timestamp
            if stamp is not None:
                pose = PoseEstimate(stamp, tuple(bridge.global_pose), self.map_id,
                                    bridge.map_match_verified)
        scan = None
        if lidar is not None and lidar.scan_stamp is not None:
            scan = RangeScan(lidar.scan_stamp, tuple(lidar.planar_angles),
                             tuple(lidar.planar_ranges), lidar.planar_max_range)
        velocity = (0., 0., 0.) if bridge is None else tuple(bridge.navigation_velocity)
        return NavigationFrame(RobotState(simulation_time, self.sequence, True, enabled, emergency),
                               pose, scan, velocity, 'simulation')
