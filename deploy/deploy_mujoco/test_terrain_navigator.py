#!/usr/bin/env python3
import numpy as np

from terrain_navigator import TerrainNavigator


class Grid:
    def __init__(self):
        self.x_values = np.arange(0.25, 1.05, 0.10)
        self.y_values = np.arange(-0.5, 0.51, 0.10)
        self.elevation = np.zeros(
            (len(self.x_values), len(self.y_values)), dtype=np.float32
        )

    def sample_corridor(self, angle, x_values, y_values):
        # Synthetic side corridors are flat; geometry-level tests exercise the
        # real angled ray scanner.
        return np.zeros((len(x_values), len(y_values)), dtype=np.float32)

    def planar_clearance(self, center_angle=0.0, half_angle=np.pi):
        return 2.0

    def translation_clearance(
        self, center_angle, half_width=0.23, body_half_length=0.39
    ):
        return 2.0 - body_half_length


def main():
    navigator = TerrainNavigator(
        {"max_step_up": 0.18, "max_drop": 0.15, "hazard_fraction": 0.15}
    )

    stairs = Grid()
    stairs.elevation[stairs.x_values >= 0.45] = 0.08
    stairs.elevation[stairs.x_values >= 0.65] = 0.16
    stairs.elevation[stairs.x_values >= 0.85] = 0.24
    assert navigator.analyze(stairs)[0] == "CLEAR"

    narrow_road = Grid()
    narrow_road.elevation[:, np.abs(narrow_road.y_values) > 0.25] = np.nan
    assert navigator.analyze(narrow_road)[0] == "CLEAR"

    wall = Grid()
    wall.elevation[wall.x_values >= 0.55] = 0.35
    wall_result = navigator.analyze(wall)
    assert wall_result[0] == "OBSTACLE", wall_result
    wall.sample_corridor = lambda angle, x_values, y_values: np.full(
        (len(x_values), len(y_values)), 0.35, dtype=np.float32
    )

    cliff = Grid()
    cliff.elevation[cliff.x_values >= 0.55] = -0.40
    cliff_result = navigator.analyze(cliff)
    assert cliff_result[0] == "DROP", cliff_result

    no_return = Grid()
    no_return.elevation[no_return.x_values >= 0.55] = np.nan
    assert navigator.analyze(no_return)[0] == "DROP"

    command = navigator.update(np.array([0.8, 0.0, 0.0]), wall, 0.0)
    assert navigator.state == "AVOID_OBSTACLE"
    assert command[0] == 0.0 and abs(command[2]) > 0.0
    assert abs(np.degrees(navigator.target_relative_angle)) <= 10.1

    clear = Grid()
    resumed = None
    for _ in range(navigator.clear_scans_required):
        resumed = navigator.update(
            np.array([0.8, 0.0, 0.0]), clear, navigator.target_yaw
        )
    assert navigator.state == "FORWARD"
    assert resumed[0] == 0.8 and resumed[2] == 0.0

    # A failed in-place turn may leave the robot in BLOCKED even though the
    # narrow corridor directly ahead is traversable.  A new straight-forward
    # command must release that stale turn lock.
    blocked_navigator = TerrainNavigator({"turning_radius": 0.43})
    tight_clear = Grid()
    tight_clear.planar_clearance = lambda center_angle=0.0, half_angle=np.pi: 0.40
    blocked_navigator.state = "BLOCKED"
    released = blocked_navigator.update(
        np.array([0.8, 0.0, 0.0]), tight_clear, 0.0
    )
    assert blocked_navigator.state == "FORWARD"
    assert released[0] == 0.8 and released[2] == 0.0

    # Keep the lock when the forward corridor itself contains a real hazard.
    blocked_wall_navigator = TerrainNavigator(
        {"turning_radius": 0.43, "hazard_fraction": 0.15}
    )
    blocked_wall = Grid()
    blocked_wall.elevation[blocked_wall.x_values >= 0.55] = 0.35
    blocked_wall.sample_corridor = lambda angle, x_values, y_values: np.full(
        (len(x_values), len(y_values)), 0.35, dtype=np.float32
    )
    blocked_wall.planar_clearance = (
        lambda center_angle=0.0, half_angle=np.pi: 0.40
    )
    blocked_wall_navigator.state = "BLOCKED"
    still_blocked = blocked_wall_navigator.update(
        np.array([0.8, 0.0, 0.0]), blocked_wall, 0.0
    )
    assert blocked_wall_navigator.state == "BLOCKED"
    assert np.allclose(still_blocked, 0.0)

    # A blocked pivot can still have a shallow body-width route.  Follow that
    # measured corridor while moving instead of inflating furniture by the
    # full in-place turning radius.
    blocked_arc_navigator = TerrainNavigator(
        {"turning_radius": 0.43, "hazard_fraction": 0.15}
    )
    blocked_arc = Grid()
    blocked_arc.elevation[blocked_arc.x_values >= 0.55] = 0.35
    blocked_arc.planar_clearance = (
        lambda center_angle=0.0, half_angle=np.pi: (
            0.48 if center_angle > 0.5 else 0.15 if center_angle < -0.5 else 0.30
        )
    )
    blocked_arc_navigator.state = "BLOCKED"
    arc_command = blocked_arc_navigator.update(
        np.array([0.8, 0.0, 0.0]), blocked_arc, 0.0
    )
    assert blocked_arc_navigator.state == "STEER_FORWARD"
    assert arc_command[0] > 0.0 and arc_command[1] > 0.0
    assert abs(arc_command[2]) > 0.0

    # Clear forward space still needs active centring: if the right wall is
    # closer, positive body-Y moves the dog left before physical contact.
    centered_navigator = TerrainNavigator({})
    centered_grid = Grid()
    centered_grid.planar_clearance = (
        lambda center_angle=0.0, half_angle=np.pi: (
            0.48 if center_angle > 0.5 else 0.15 if center_angle < -0.5 else 2.0
        )
    )
    center_left = centered_navigator.update(
        np.array([1.0, 0.0, 0.0]), centered_grid, 0.0
    )
    assert center_left[0] == centered_navigator.centering_forward_speed
    assert center_left[1] > 0.0 and center_left[2] == 0.0

    # The RL gait can drift right even while commanded forward.  Negative
    # measured body-Y must strengthen the left correction, including when the
    # body is momentarily centred between two nearby walls.
    center_left_with_drift = centered_navigator.update(
        np.array([1.0, 0.0, 0.0]),
        centered_grid,
        0.0,
        lateral_velocity=-0.13,
    )
    assert center_left_with_drift[1] > center_left[1]
    assert center_left_with_drift[1] <= centered_navigator.centering_max_lateral

    velocity_navigator = TerrainNavigator({})
    equal_corridor = Grid()
    equal_corridor.planar_clearance = (
        lambda center_angle=0.0, half_angle=np.pi: 0.40
    )
    drift_only = velocity_navigator.update(
        np.array([1.0, 0.0, 0.0]),
        equal_corridor,
        0.0,
        lateral_velocity=-0.10,
    )
    assert drift_only[1] > 0.0

    print("stairs=CLEAR")
    print("narrow_road=CLEAR_FORWARD")
    print(f"wall={wall_result[0]} at {wall_result[1]:.2f}m")
    print(f"cliff={cliff_result[0]} at {cliff_result[1]:.2f}m")
    print("missing_ground=DROP")
    print(f"avoidance_command={command.tolist()}")
    print(f"resume_command={resumed.tolist()}")
    print(f"blocked_clear_release={released.tolist()}")
    print(f"blocked_wall_command={still_blocked.tolist()}")
    print(f"blocked_forward_arc={arc_command.tolist()}")
    print(f"corridor_centering={center_left.tolist()}")
    print(f"centering_with_right_drift={center_left_with_drift.tolist()}")
    print(f"drift_only_correction={drift_only.tolist()}")


if __name__ == "__main__":
    main()
