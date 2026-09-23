#!/usr/bin/env python3
"""Small deterministic checks for the high-level navigation runtime."""

from pathlib import Path

import numpy as np

from high_level_nav_policy import (
    OBSERVATION_SIZE,
    HighLevelNavigationPolicy,
    apply_gait_command_envelope,
    build_navigation_observation,
)


class FakeLidar:
    planar_ranges = np.full(72, 2.0, dtype=np.float32)
    planar_max_range = 2.0


def main():
    observation = build_navigation_observation(
        target_world=[2.0, 1.0],
        position_world=[0.0, 0.0],
        yaw=0.2,
        local_velocity=[0.1, 0.0, 0.0],
        previous_command=[0.0, 0.0, 0.0],
        planar_ranges=FakeLidar.planar_ranges,
        planar_max_range=FakeLidar.planar_max_range,
    )
    assert observation.shape == (OBSERVATION_SIZE,)
    assert np.isfinite(observation).all()

    straight = apply_gait_command_envelope([0.75, 0.2, 0.0], 0.9)
    np.testing.assert_allclose(straight, [0.75, 0.2, 0.0])
    hard_turn = apply_gait_command_envelope([0.75, 0.2, 0.9], 0.9)
    np.testing.assert_allclose(hard_turn, [0.2625, 0.07, 0.9], atol=1.0e-6)

    deploy_root = Path(__file__).resolve().parents[1]
    policy = HighLevelNavigationPolicy(
        {
            "enabled": True,
            "policy_path": deploy_root / "pre_train/go2/high_level_nav.pt",
            "max_speed": 0.75,
            "max_lateral_speed": 0.30,
            "max_yaw_rate": 0.90,
            "turn_speed_reduction": 0.65,
        }
    )
    command = policy.command(
        target=[2.0, 0.5],
        position=[0.0, 0.0],
        yaw=0.0,
        local_velocity=[0.0, 0.0, 0.0],
        lidar=FakeLidar(),
    )
    assert command.shape == (3,)
    assert 0.0 <= command[0] <= 0.75
    assert abs(command[1]) <= 0.30
    assert abs(command[2]) <= 0.90
    # A side/rear goal requests a genuine pivot so the safety layer can
    # release the turning envelope instead of selecting a competing arc.
    for sign in (-1, 1):
        policy.reset()
        for degrees in (100, 40):
            angle = np.deg2rad(sign * degrees)
            turning = policy.command(
                [2 * np.cos(angle), 2 * np.sin(angle)], [0, 0], 0,
                [0, 0, 0], FakeLidar(),
            )
            np.testing.assert_array_equal(turning[:2], [0, 0])
            assert sign * turning[2] > 0
        angle = np.deg2rad(sign * 10)
        policy.command(
            [2 * np.cos(angle), 2 * np.sin(angle)], [0, 0], 0,
            [0, 0, 0], FakeLidar(),
        )
        assert not policy.reorienting
    policy.reset()
    assert not policy.reorienting
    np.testing.assert_array_equal(policy.previous_command, [0, 0, 0])
    print(f"high-level navigation checks passed: command={command}")


if __name__ == "__main__":
    main()
