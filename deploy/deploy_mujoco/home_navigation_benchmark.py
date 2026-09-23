#!/usr/bin/env python3
"""Headless, deterministic furnished-home navigation regression benchmark."""

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml

from goal_navigator import GoalNavigator
from high_level_nav_policy import HighLevelNavigationPolicy
from lidar_heightmap import LidarHeightMap
from perceptive_observation import PerceptiveObservationBuilder
from terrain_navigator import TerrainNavigator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "deploy/deploy_mujoco/configs/go2_home.yaml"
DEFAULT_SCENARIOS = ROOT / "deploy/deploy_mujoco/configs/home_benchmark.yaml"
DEFAULT_BASELINE = ROOT / "benchmarks/home_navigation_baseline.json"


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
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def strongest_environment_contact(model, data):
    """Return strongest robot/furniture contact, excluding normal flooring."""
    ignored = {"floor", "living_rug"}
    strongest = None
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if (body1 == 0) == (body2 == 0):
            continue
        environment_geom = geom1 if body1 == 0 else geom2
        robot_geom = geom2 if body1 == 0 else geom1
        environment_name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom
            )
            or f"geom{environment_geom}"
        )
        if environment_name in ignored:
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        magnitude = float(np.linalg.norm(force[:3]))
        robot_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
            or f"geom{robot_geom}"
        )
        if strongest is None or magnitude > strongest[0]:
            strongest = (magnitude, robot_name, environment_name)
    return strongest


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(
        sum(np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points)))
    )


class HomeTrial:
    def __init__(self, config, benchmark_config):
        self.config = config
        self.benchmark_config = benchmark_config
        self.model_path = Path(
            config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT))
        )
        self.policy_path = Path(
            config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT))
        )

    def run(self, name, scenario):
        model = mujoco.MjModel.from_xml_path(str(self.model_path))
        data = mujoco.MjData(model)
        model.opt.timestep = float(self.config["simulation_dt"])
        default_angles = np.asarray(self.config["default_angles"], dtype=np.float32)
        start_x, start_y, start_yaw = map(float, scenario["start"])
        data.qpos[:7] = [
            start_x,
            start_y,
            0.30,
            math.cos(start_yaw / 2.0),
            0.0,
            0.0,
            math.sin(start_yaw / 2.0),
        ]
        data.qpos[7:] = default_angles
        mujoco.mj_forward(model, data)

        lidar = LidarHeightMap(model, data, self.config["lidar"])
        lidar.scan()
        scan_steps = max(1, int(lidar.scan_interval / model.opt.timestep))
        navigator = TerrainNavigator(self.config["navigation"])
        goal_navigator = GoalNavigator(model, data, self.config["goal_navigation"])
        high_config = dict(self.config["high_level_navigation"])
        high_config["policy_path"] = str(high_config["policy_path"]).replace(
            "{LEGGED_GYM_ROOT_DIR}", str(ROOT)
        )
        high_level = HighLevelNavigationPolicy(high_config)
        low_level = torch.jit.load(str(self.policy_path), map_location="cpu").eval()
        privileged_builder = PerceptiveObservationBuilder(
            model,
            model.opt.timestep * int(self.config["control_decimation"]),
        )

        kps = np.asarray(self.config["kps"], dtype=np.float32)
        kds = np.asarray(self.config["kds"], dtype=np.float32)
        cmd_scale = np.asarray(self.config["cmd_scale"], dtype=np.float32)
        decimation = int(self.config["control_decimation"])
        num_actions = int(self.config["num_actions"])
        observation = np.zeros(int(self.config["num_obs"]), dtype=np.float32)
        action = np.zeros(num_actions, dtype=np.float32)
        target = default_angles.copy()
        torque = np.zeros(num_actions, dtype=np.float32)
        command = np.zeros(3, dtype=np.float32)

        mj_names = self.config["mujoco_joint_names"]
        policy_names = self.config["model_joint_names"]
        model_to_mj = [policy_names.index(joint) for joint in mj_names]
        mj_to_model = [mj_names.index(joint) for joint in policy_names]

        settle_time = float(self.benchmark_config.get("settle_time", 1.5))
        timeout = float(scenario["timeout"])
        contact_threshold = float(
            self.benchmark_config.get("contact_force_threshold", 5.0)
        )
        max_steps = int((settle_time + timeout) / model.opt.timestep)
        goal_started = False
        planned_length = 0.0
        route_points = [np.asarray(data.qpos[:2], dtype=np.float64).copy()]
        progress_samples = []
        contact_episodes = 0
        contact_active = False
        maximum_contact_force = 0.0
        strongest_pair = None
        blocked_time = 0.0
        fell = False
        started_wall_time = time.perf_counter()

        for counter in range(max_steps):
            if not goal_started and data.time >= settle_time:
                goal_started = True
                goal_navigator.set_goal(
                    scenario["goal"], data.qpos[:2], data.time, name=name
                )
                planned_length = path_length(goal_navigator.path)
                route_points = [np.asarray(data.qpos[:2], dtype=np.float64).copy()]

            if counter % decimation == 0:
                local_velocity = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
                local_angular = quat_rotate_inverse(data.qpos[3:7], data.qvel[3:6])
                if goal_navigator.active:
                    goal_navigator.update(
                        data.qpos[:2], yaw_from_quaternion(data.qpos[3:7]), data.time
                    )
                if goal_navigator.active:
                    waypoint = goal_navigator.path[
                        min(goal_navigator.waypoint_index, len(goal_navigator.path) - 1)
                    ]
                    requested = high_level.command(
                        waypoint,
                        data.qpos[:2],
                        yaw_from_quaternion(data.qpos[3:7]),
                        [local_velocity[0], local_velocity[1], local_angular[2]],
                        lidar,
                    )
                else:
                    requested = np.zeros(3, dtype=np.float32)
                command = navigator.update(
                    requested,
                    lidar,
                    yaw_from_quaternion(data.qpos[3:7]),
                    autonomous=goal_navigator.active,
                    lateral_velocity=float(local_velocity[1]),
                    goal_distance=(
                        goal_navigator.last_distance if goal_navigator.active else None
                    ),
                )
                if navigator.state == "BLOCKED":
                    blocked_time += model.opt.timestep * decimation

            torque = (target - data.qpos[7:]) * kps - data.qvel[6:] * kds
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if counter % scan_steps == 0:
                lidar.scan()

            contact = strongest_environment_contact(model, data)
            force = 0.0 if contact is None else contact[0]
            above_threshold = force >= contact_threshold
            if above_threshold and not contact_active:
                contact_episodes += 1
            contact_active = above_threshold
            if force > maximum_contact_force:
                maximum_contact_force = force
                strongest_pair = None if contact is None else [contact[1], contact[2]]

            if counter % decimation == 0:
                route_points.append(np.asarray(data.qpos[:2], dtype=np.float64).copy())
                if goal_started:
                    progress_samples.append(
                        (float(data.time), float(goal_navigator.last_distance))
                    )
                qj = (data.qpos[7:] - default_angles) * float(
                    self.config["dof_pos_scale"]
                )
                dqj = data.qvel[6:] * float(self.config["dof_vel_scale"])
                observation[:3] = data.qvel[3:6] * float(
                    self.config["ang_vel_scale"]
                )
                observation[3:6] = gravity_orientation(data.qpos[3:7])
                observation[6:9] = command * cmd_scale
                observation[9 : 9 + num_actions] = qj[mj_to_model]
                observation[9 + num_actions : 9 + 2 * num_actions] = dqj[
                    mj_to_model
                ]
                observation[9 + 2 * num_actions : 9 + 3 * num_actions] = action[
                    mj_to_model
                ]
                local_velocity = quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
                privileged = privileged_builder.build(
                    data,
                    observation,
                    local_velocity,
                    torque,
                    lidar,
                    mj_to_model,
                )
                with torch.inference_mode():
                    result = low_level(
                        torch.from_numpy(observation).unsqueeze(0),
                        torch.from_numpy(privileged).unsqueeze(0),
                    )
                if isinstance(result, tuple):
                    result = result[0]
                action = result.detach().cpu().numpy().squeeze()[model_to_mj]
                target = action * float(self.config["action_scale"]) + default_angles

            if data.time > 1.0:
                upright = float(gravity_orientation(data.qpos[3:7])[2])
                if float(data.qpos[2]) < 0.18 or upright > -0.15:
                    fell = True
                    break
            if goal_started and (goal_navigator.reached or goal_navigator.failed):
                break

        wall_time = time.perf_counter() - started_wall_time
        travelled = path_length(route_points)
        route_time = max(0.0, float(data.time) - settle_time)
        ratio = travelled / max(planned_length, 1.0e-6)
        stuck = False
        stuck_window = float(self.benchmark_config.get("stuck_window", 5.0))
        if goal_navigator.active and progress_samples:
            cutoff = progress_samples[-1][0] - stuck_window
            recent = [distance for stamp, distance in progress_samples if stamp >= cutoff]
            if len(recent) > 1:
                stuck = max(recent) - min(recent) < float(
                    self.benchmark_config.get("stuck_distance", 0.05)
                )

        limits = {
            "max_contact_episodes": int(scenario["max_contact_episodes"]),
            "max_contact_force": float(scenario["max_contact_force"]),
            "max_path_ratio": float(scenario["max_path_ratio"]),
            "timeout": timeout,
        }
        checks = {
            "goal_reached": bool(goal_navigator.reached),
            "did_not_fall": not fell,
            "not_stuck": not stuck,
            "contact_episodes": contact_episodes <= limits["max_contact_episodes"],
            "contact_force": maximum_contact_force <= limits["max_contact_force"],
            "path_ratio": ratio <= limits["max_path_ratio"],
            "within_timeout": route_time <= timeout + model.opt.timestep,
        }
        return {
            "name": name,
            "description": scenario.get("description", ""),
            "passed": all(checks.values()),
            "checks": checks,
            "goal_reached": bool(goal_navigator.reached),
            "planner_failed": bool(goal_navigator.failed),
            "fell": fell,
            "stuck": stuck,
            "simulation_time_s": round(route_time, 3),
            "wall_time_s": round(wall_time, 3),
            "planned_path_m": round(planned_length, 3),
            "travelled_path_m": round(travelled, 3),
            "path_ratio": round(ratio, 3),
            "contact_episodes": contact_episodes,
            "max_contact_force_n": round(maximum_contact_force, 3),
            "strongest_contact": strongest_pair,
            "blocked_time_s": round(blocked_time, 3),
            "final_position": [round(float(value), 3) for value in data.qpos[:3]],
            "effective_goal": [
                round(float(value), 3) for value in goal_navigator.effective_goal
            ],
            "limits": limits,
        }


def markdown_report(report):
    lines = [
        "# Go2 家庭导航回归报告",
        "",
        f"- 严格质量门：{'通过' if report['quality_passed'] else '失败'}",
        f"- 相对保存基线：{'无退化' if report['regression_passed'] else '发生退化'}",
        f"- 通过场景：{report['passed_count']}/{report['scenario_count']}",
        f"- 总墙钟时间：{report['wall_time_s']:.1f} 秒",
        "",
        "| 场景 | 结果 | 到达 | 仿真时间 | 路程/规划 | 碰撞次数 | 峰值力 | 阻塞时间 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["scenarios"]:
        lines.append(
            "| {name} | {result} | {reached} | {time:.1f}s | {travel:.2f}/{planned:.2f}m "
            "| {contacts} | {force:.1f}N | {blocked:.1f}s |".format(
                name=item["name"],
                result="PASS" if item["passed"] else "FAIL",
                reached="是" if item["goal_reached"] else "否",
                time=item["simulation_time_s"],
                travel=item["travelled_path_m"],
                planned=item["planned_path_m"],
                contacts=item["contact_episodes"],
                force=item["max_contact_force_n"],
                blocked=item["blocked_time_s"],
            )
        )
    failures = [item for item in report["scenarios"] if not item["passed"]]
    if failures:
        lines.extend(["", "## 未通过项", ""])
        for item in failures:
            failed_checks = [key for key, value in item["checks"].items() if not value]
            lines.append(f"- `{item['name']}`：{', '.join(failed_checks)}")
    if report["regressions"]:
        lines.extend(["", "## 相对基线的退化", ""])
        lines.extend(f"- {message}" for message in report["regressions"])
    lines.extend(
        [
            "",
            "报告中的碰撞只统计机器人与家具/墙体的接触；地板和地毯接触已排除。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_with_baseline(results, baseline):
    """Return human-readable regressions while allowing genuine improvements."""
    previous = baseline.get("scenarios", {}) if baseline else {}
    regressions = []
    for current in results:
        old = previous.get(current["name"])
        if old is None:
            continue
        name = current["name"]
        if bool(old.get("goal_reached")) and not current["goal_reached"]:
            regressions.append(f"`{name}`：原本可到达，现在未到达")
        if not bool(old.get("fell", False)) and current["fell"]:
            regressions.append(f"`{name}`：新增跌倒")
        old_contacts = int(old.get("contact_episodes", 0))
        if current["contact_episodes"] > old_contacts:
            regressions.append(
                f"`{name}`：碰撞次数 {old_contacts} → {current['contact_episodes']}"
            )
        old_force = float(old.get("max_contact_force_n", 0.0))
        force_limit = max(old_force + 10.0, old_force * 1.25)
        if current["max_contact_force_n"] > force_limit:
            regressions.append(
                f"`{name}`：峰值碰撞力 {old_force:.1f}N → "
                f"{current['max_contact_force_n']:.1f}N"
            )
        old_time = float(old.get("simulation_time_s", 0.0))
        if current["goal_reached"] and old_time > 0.0:
            time_limit = old_time * 1.35 + 1.0
            if current["simulation_time_s"] > time_limit:
                regressions.append(
                    f"`{name}`：到达时间 {old_time:.1f}s → "
                    f"{current['simulation_time_s']:.1f}s"
                )
        old_path = float(old.get("travelled_path_m", 0.0))
        if current["goal_reached"] and old_path > 0.0:
            path_limit = old_path * 1.25 + 0.25
            if current["travelled_path_m"] > path_limit:
                regressions.append(
                    f"`{name}`：实际路径 {old_path:.2f}m → "
                    f"{current['travelled_path_m']:.2f}m"
                )
    return regressions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Saved result used to detect relative regressions",
    )
    parser.add_argument("--only", nargs="*", help="Run only named scenarios")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    benchmark_config = yaml.safe_load(args.scenarios.read_text())
    scenarios = benchmark_config["scenarios"]
    selected = list(scenarios) if not args.only else args.only
    unknown = sorted(set(selected) - set(scenarios))
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}")

    started = time.perf_counter()
    runner = HomeTrial(config, benchmark_config)
    results = []
    for name in selected:
        print(f"[RUN] {name}: {scenarios[name].get('description', '')}", flush=True)
        result = runner.run(name, scenarios[name])
        results.append(result)
        print(
            f"[{'PASS' if result['passed'] else 'FAIL'}] {name}: "
            f"reached={result['goal_reached']} time={result['simulation_time_s']:.1f}s "
            f"contacts={result['contact_episodes']} "
            f"max_force={result['max_contact_force_n']:.1f}N",
            flush=True,
        )

    baseline = (
        json.loads(args.baseline.read_text())
        if args.baseline and args.baseline.is_file()
        else {}
    )
    regressions = compare_with_baseline(results, baseline)
    quality_passed = all(item["passed"] for item in results)
    report = {
        "schema_version": 1,
        "passed": quality_passed and not regressions,
        "quality_passed": quality_passed,
        "regression_passed": not regressions,
        "regressions": regressions,
        "baseline": str(args.baseline) if args.baseline else None,
        "scenario_count": len(results),
        "passed_count": sum(item["passed"] for item in results),
        "wall_time_s": round(time.perf_counter() - started, 3),
        "policy": str(runner.policy_path),
        "high_level_policy": str(
            config["high_level_navigation"]["policy_path"]
        ).replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)),
        "scenarios": results,
    }
    rendered = markdown_report(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered)
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
