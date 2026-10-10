"""Runtime adapter for the learned high-level navigation velocity policy."""

from pathlib import Path

import numpy as np
import torch


LIDAR_RAY_COUNT = 24
OBSERVATION_SIZE = 33


def apply_gait_command_envelope(command, max_yaw_rate, turn_speed_reduction=0.65):
    """Reduce translation while turning sharply.

    The low-level Go2 gait can track either a fast translation or a fast yaw
    command, but combining both makes the rear legs sweep outside the path
    assumed by the point-like high-level training world.  This is a command
    feasibility limit, not extra obstacle inflation: straight narrow-passage
    motion is unchanged.
    """
    result = np.asarray(command, dtype=np.float32).copy()
    if max_yaw_rate <= 0.0:
        return result
    turn_fraction = float(np.clip(abs(result[2]) / max_yaw_rate, 0.0, 1.0))
    translation_scale = 1.0 - float(turn_speed_reduction) * turn_fraction
    result[:2] *= max(0.0, translation_scale)
    return result


def build_navigation_observation(
    target_world,
    position_world,
    yaw,
    local_velocity,
    previous_command,
    planar_ranges,
    planar_max_range,
):
    """Build the sensor-only observation shared by training and deployment."""
    target_world = np.asarray(target_world, dtype=np.float32)[:2]
    position_world = np.asarray(position_world, dtype=np.float32)[:2]
    delta = target_world - position_world
    cy, sy = np.cos(float(yaw)), np.sin(float(yaw))
    local_goal = np.array(
        [cy * delta[0] + sy * delta[1], -sy * delta[0] + cy * delta[1]],
        dtype=np.float32,
    )
    distance = float(np.linalg.norm(local_goal))
    bearing = float(np.arctan2(local_goal[1], local_goal[0]))

    ranges = np.asarray(planar_ranges, dtype=np.float32).reshape(-1)
    if ranges.size < LIDAR_RAY_COUNT:
        raise ValueError("planar LiDAR has too few rays")
    indices = np.linspace(
        0, ranges.size, LIDAR_RAY_COUNT, endpoint=False, dtype=np.int32
    )
    lidar = np.clip(ranges[indices] / float(planar_max_range), 0.0, 1.0)
    velocity = np.asarray(local_velocity, dtype=np.float32)[:3]
    previous = np.asarray(previous_command, dtype=np.float32)[:3]
    observation = np.concatenate(
        (
            np.array(
                [min(distance / 4.0, 1.0), np.cos(bearing), np.sin(bearing)],
                dtype=np.float32,
            ),
            np.clip(velocity / np.array([0.8, 0.4, 1.2]), -1.5, 1.5),
            np.clip(previous / np.array([0.8, 0.4, 1.2]), -1.0, 1.0),
            lidar,
        )
    ).astype(np.float32)
    if observation.size != OBSERVATION_SIZE:
        raise RuntimeError(
            f"navigation observation has {observation.size} values, "
            f"expected {OBSERVATION_SIZE}"
        )
    return observation


class HighLevelNavigationPolicy:
    """Convert goal and planar-LiDAR observations into body velocity commands."""

    def __init__(self, config):
        self.enabled = bool(config.get("enabled", False))
        self.policy_path = Path(str(config.get("policy_path", ""))).expanduser()
        self.max_speed = float(config.get("max_speed", 0.75))
        self.max_lateral_speed = float(config.get("max_lateral_speed", 0.30))
        self.max_yaw_rate = float(config.get("max_yaw_rate", 0.90))
        self.turn_speed_reduction = float(
            config.get("turn_speed_reduction", 0.65)
        )
        self.previous_command = np.zeros(3, dtype=np.float32)
        self.reorienting = False
        self.policy = None
        if self.enabled:
            if not self.policy_path.is_file():
                raise RuntimeError(
                    f"High-level navigation policy not found: {self.policy_path}"
                )
            self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
            self.policy.eval()
            print(f"High-level navigation PPO loaded: {self.policy_path}")

    def reset(self):
        self.previous_command.fill(0.0)
        self.reorienting = False

    def command(self, target, position, yaw, local_velocity, lidar):
        if not self.enabled or self.policy is None:
            return None
        delta = np.asarray(target)[:2] - np.asarray(position)[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 0.40:
            # The PPO's training success radius is too coarse for docking.
            # At short range its yaw/lateral actions can orbit the target.
            # Resolve the remaining sensor-pose error with a bounded body
            # translation, retaining the downstream terrain safety filter
            # and the same RL locomotion policy. No simulator pose is used.
            cy, sy = np.cos(yaw), np.sin(yaw)
            local_error = np.array([
                cy * delta[0] + sy * delta[1],
                -sy * delta[0] + cy * delta[1],
            ])
            command = np.zeros(3, dtype=np.float32)
            command[:2] = np.clip(
                0.8 * local_error - 0.15 * np.asarray(local_velocity)[:2],
                [-min(0.20, self.max_speed), -min(0.20, self.max_lateral_speed)],
                [min(0.20, self.max_speed), min(0.20, self.max_lateral_speed)],
            )
            self.reorienting = False
            self.previous_command[:] = command
            return command
        heading_error = float(np.arctan2(
            np.sin(np.arctan2(delta[1], delta[0]) - yaw),
            np.cos(np.arctan2(delta[1], delta[0]) - yaw),
        ))
        # A small positive PPO forward speed must not disguise a requested
        # pivot as a forward arc. The terrain navigator can make room for a
        # pure turn using its measured body envelope. Hysteresis prevents
        # repeatedly switching between pivot and translation near the limit.
        if abs(heading_error) > np.deg2rad(60.0):
            self.reorienting = True
        elif abs(heading_error) < np.deg2rad(15.0):
            self.reorienting = False
        if self.reorienting:
            command = np.array([
                0.0, 0.0,
                np.clip(1.8 * heading_error, -self.max_yaw_rate, self.max_yaw_rate),
            ], dtype=np.float32)
            self.previous_command[:] = command
            return command
        observation = build_navigation_observation(
            target,
            position,
            yaw,
            local_velocity,
            self.previous_command,
            lidar.planar_ranges,
            lidar.planar_max_range,
        )
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(observation).unsqueeze(0))
        action = np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        action = np.clip(action[:3], -1.0, 1.0)
        command = np.array(
            [
                0.5 * (action[0] + 1.0) * self.max_speed,
                action[1] * self.max_lateral_speed,
                action[2] * self.max_yaw_rate,
            ],
            dtype=np.float32,
        )
        distance = float(
            np.linalg.norm(np.asarray(target)[:2] - np.asarray(position)[:2])
        )
        command[0] *= np.clip(distance / 0.50, 0.15, 1.0)
        command = apply_gait_command_envelope(
            command,
            self.max_yaw_rate,
            self.turn_speed_reduction,
        )
        self.previous_command[:] = command
        return command
