#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
import yaml

from lidar_heightmap import LidarHeightMap
from perceptive_observation import PerceptiveObservationBuilder
from terrain_navigator import TerrainNavigator


ROOT = Path(__file__).resolve().parents[2]


def gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qw = q[0]
    qv = q[1:]
    return (
        v * (2.0 * qw * qw - 1.0)
        - np.cross(qv, v) * qw * 2.0
        + qv * np.dot(qv, v) * 2.0
    )


def yaw_from_quaternion(q):
    qw, qx, qy, qz = q
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def run_trial(speed, duration, grid_output=None, xml_name=None):
    config_path = ROOT / "deploy/deploy_mujoco/configs/go2_stairs.yaml"
    config = yaml.safe_load(config_path.read_text())
    policy_path = Path(config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)))
    xml_path = Path(config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)))
    if xml_name:
        xml_path = ROOT / "resources/robots/go2" / xml_name

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = config["simulation_dt"]
    mujoco.mj_forward(model, data)
    lidar = LidarHeightMap(model, data, config["lidar"])
    lidar.scan()
    lidar_scan_steps = max(1, int(lidar.scan_interval / model.opt.timestep))

    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    kps = np.asarray(config["kps"], dtype=np.float32)
    kds = np.asarray(config["kds"], dtype=np.float32)
    default_angles = np.asarray(config["default_angles"], dtype=np.float32)
    cmd_scale = np.asarray(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    decimation = config["control_decimation"]
    perceptive_policy = bool(config.get("perceptive_policy", False))
    privileged_builder = PerceptiveObservationBuilder(
        model, model.opt.timestep * decimation
    ) if perceptive_policy else None
    navigator = TerrainNavigator(config.get("navigation", {}))

    mj_names = config["mujoco_joint_names"]
    policy_names = config["model_joint_names"]
    model_to_mj = [policy_names.index(name) for name in mj_names]
    mj_to_model = [mj_names.index(name) for name in policy_names]

    action = np.zeros(num_actions, dtype=np.float32)
    target = default_angles.copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    manual_command = np.zeros(3, dtype=np.float32)
    stationary_updates = 0
    hold_active = False
    hold_target = target.copy()
    hold_start_position = None
    hold_settle_updates = max(1, int(0.5 / (model.opt.timestep * decimation)))
    hold_kps = np.full_like(kps, 40.0)
    hold_kds = np.full_like(kds, 1.5)
    max_x = float(data.qpos[0])
    max_z = float(data.qpos[2])
    fell = False
    first_detection = None
    first_policy_terrain = None
    max_precontact_action_delta = 0.0
    max_precontact_joint_delta = 0.0
    avoidance_events = 0
    previous_navigation_state = navigator.state
    navigation_transitions = []
    milestones = {1.2: None, 1.8: None, 2.4: None, 3.0: None}

    total_steps = int(duration / model.opt.timestep)
    for counter in range(total_steps):
        sim_time = counter * model.opt.timestep
        if 1.5 <= sim_time < duration - 1.0:
            manual_command[:] = (speed, 0.0, 0.0)
        else:
            manual_command[:] = 0.0

        if counter % lidar_scan_steps == 0:
            lidar.scan()
            if lidar.stairs_detected and first_detection is None:
                first_detection = (sim_time, lidar.nearest_rise, lidar.max_rise)
                if grid_output:
                    output_path = Path(grid_output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(lidar.image, cv2.COLOR_RGB2BGR),
                    )
        command[:] = navigator.update(
            manual_command, lidar, yaw_from_quaternion(data.qpos[3:7])
        )
        if navigator.state != previous_navigation_state:
            if navigator.state != "FORWARD":
                avoidance_events += 1
            navigation_transitions.append(
                (
                    sim_time,
                    navigator.state,
                    float(data.qpos[0]),
                    float(data.qpos[1]),
                    float(yaw_from_quaternion(data.qpos[3:7])),
                    navigator.hazard,
                    "LEFT" if navigator.turn_direction > 0 else "RIGHT",
                )
            )
            previous_navigation_state = navigator.state

        active_kps = hold_kps if hold_active else kps
        active_kds = hold_kds if hold_active else kds
        torque = (target - data.qpos[7:]) * active_kps - data.qvel[6:] * active_kds
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)

        if counter % decimation == 0:
            if np.linalg.norm(command) < 1e-4:
                stationary_updates += 1
                if stationary_updates >= hold_settle_updates and not hold_active:
                    hold_target = default_angles.copy()
                    hold_active = True
                    hold_start_position = data.qpos[:2].copy()
            else:
                stationary_updates = 0
                hold_active = False

            if hold_active:
                target = hold_target
            else:
                qj = (data.qpos[7:] - default_angles) * config["dof_pos_scale"]
                dqj = data.qvel[6:] * config["dof_vel_scale"]
                obs[:3] = data.qvel[3:6] * config["ang_vel_scale"]
                obs[3:6] = gravity_orientation(data.qpos[3:7])
                obs[6:9] = command * cmd_scale
                obs[9 : 9 + num_actions] = qj[mj_to_model]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj[mj_to_model]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[mj_to_model]
                with torch.inference_mode():
                    if perceptive_policy:
                        local_velocity = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
                        privileged = privileged_builder.build(
                            data, obs, local_velocity, torque, lidar, mj_to_model
                        )
                        result = policy(
                            torch.from_numpy(obs).unsqueeze(0),
                            torch.from_numpy(privileged).unsqueeze(0),
                        )
                        # Counterfactual: same robot state, but tell the network
                        # that all 187 terrain samples are flat. This isolates
                        # the causal contribution of exteroception before impact.
                        flat_privileged = privileged.copy()
                        flat_height = np.clip(
                            float(data.qpos[2]) - 0.5 - lidar.baseline, -1.0, 1.0
                        ) * 2.5
                        flat_privileged[-187:] = flat_height
                        flat_result = policy(
                            torch.from_numpy(obs).unsqueeze(0),
                            torch.from_numpy(flat_privileged).unsqueeze(0),
                        )
                        terrain_visible = np.max(
                            lidar.policy_heights - lidar.baseline
                        ) > lidar.rise_threshold
                        if terrain_visible and first_policy_terrain is None:
                            actual_action = result[0]
                            flat_action = flat_result[0]
                            first_policy_terrain = (
                                sim_time,
                                float(data.qpos[0]),
                                float(torch.linalg.vector_norm(actual_action - flat_action)),
                            )
                        if float(data.qpos[0]) < 0.95:
                            actual_action = result[0]
                            flat_action = flat_result[0]
                            max_precontact_action_delta = max(
                                max_precontact_action_delta,
                                float(torch.linalg.vector_norm(actual_action - flat_action)),
                            )
                            max_precontact_joint_delta = max(
                                max_precontact_joint_delta,
                                float(torch.max(torch.abs(actual_action - flat_action)))
                                * config["action_scale"],
                            )
                    else:
                        result = policy(torch.from_numpy(obs).unsqueeze(0))
                if isinstance(result, tuple):
                    result = result[0]
                action = result.detach().cpu().numpy().squeeze()[model_to_mj]
                target = action * config["action_scale"] + default_angles

        x = float(data.qpos[0])
        z = float(data.qpos[2])
        max_x = max(max_x, x)
        max_z = max(max_z, z)
        for position in milestones:
            if milestones[position] is None and x >= position:
                milestones[position] = (sim_time, z)
        if sim_time > 1.0 and (z < 0.18 or gravity_orientation(data.qpos[3:7])[2] > -0.15):
            fell = True
            break

    top_reached = max_x >= 3.0 and max_z >= 0.78 and not fell
    print(f"policy={policy_path.name}")
    print(f"scene={xml_path.name}")
    print(f"speed={speed:.2f} m/s")
    print(f"sim_time={data.time:.2f} s")
    print(f"final_position=({data.qpos[0]:+.3f}, {data.qpos[1]:+.3f}, {data.qpos[2]:+.3f})")
    print(f"max_x={max_x:.3f} m")
    print(f"max_base_height={max_z:.3f} m")
    if first_detection is None:
        print("lidar_detection=none")
    else:
        print(
            f"lidar_detection=t={first_detection[0]:.2f}s, "
            f"nearest={first_detection[1]:.2f}m, rise={first_detection[2]:.2f}m"
        )
    if first_policy_terrain is None:
        print("policy_terrain_input=none")
    else:
        print(
            f"policy_terrain_input=t={first_policy_terrain[0]:.2f}s, "
            f"base_x={first_policy_terrain[1]:.3f}m, "
            f"counterfactual_action_l2={first_policy_terrain[2]:.4f}"
        )
    print(f"precontact_action_l2_max={max_precontact_action_delta:.4f}")
    print(f"precontact_joint_target_delta_max={max_precontact_joint_delta:.4f} rad")
    print(f"navigation_avoidance_events={avoidance_events}")
    for transition in navigation_transitions:
        print(
            "navigation_transition="
            f"t={transition[0]:.2f}, state={transition[1]}, "
            f"pos=({transition[2]:.2f},{transition[3]:.2f}), "
            f"yaw={transition[4]:.2f}, hazard={transition[5]}, turn={transition[6]}"
        )
    if hold_start_position is not None:
        hold_drift = np.linalg.norm(data.qpos[:2] - hold_start_position)
        print(f"drift_after_hold={hold_drift:.4f} m")
    for position, reached in milestones.items():
        if reached is None:
            print(f"x={position:.1f}: not reached")
        else:
            print(f"x={position:.1f}: t={reached[0]:.2f}s, base_z={reached[1]:.3f}m")
    print(f"fell={fell}")
    print(f"top_reached={top_reached}")
    return 0 if top_reached else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.55)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--grid-output")
    parser.add_argument("--xml", help="Override the MuJoCo scene filename.")
    args = parser.parse_args()
    return run_trial(args.speed, args.duration, args.grid_output, args.xml)


if __name__ == "__main__":
    raise SystemExit(main())
