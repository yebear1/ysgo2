"""Platform-neutral packets. All stamps use the injected monotonic clock.

ROS/device timestamps must be converted by adapters, never mixed with wall time.
Camera/IMU packets feed a localization provider; the controller consumes its
verified map pose, not pixels or simulator ground truth.
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraSample:
    stamp: float
    frame_id: str
    rgb: Any
    depth_m: Any
    intrinsics: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ImuSample:
    stamp: float
    frame_id: str
    quaternion_wxyz: Tuple[float, float, float, float]
    angular_velocity: Vector3
    acceleration: Vector3


@dataclass(frozen=True)
class PoseEstimate:
    stamp: float
    xy_yaw: Vector3
    map_id: str
    matched: bool
    frame_id: str = 'map'


@dataclass(frozen=True)
class RangeScan:
    stamp: float
    angles: Tuple[float, ...]
    ranges: Tuple[float, ...]
    max_range: float
    frame_id: str = 'base_link'


@dataclass(frozen=True)
class RobotState:
    stamp: float
    sequence: int  # Unwrapped adapter sequence, not a wrapping firmware tick.
    connected: bool
    enabled: bool  # Operator motion enable, not motor/sport-mode state.
    emergency: bool = False


@dataclass(frozen=True)
class NavigationFrame:
    robot: RobotState
    pose: Optional[PoseEstimate] = None
    scan: Optional[RangeScan] = None
    velocity_body: Vector3 = (0., 0., 0.)
    clock_id: str = 'monotonic'


@dataclass(frozen=True)
class VelocityRequest:
    stamp: float
    velocity: Vector3
    epoch: int = 0


@dataclass(frozen=True)
class Decision:
    source: str
    velocity: Vector3
    reason: str
    transport_ok: bool
    physical_stop_verified: bool = False
