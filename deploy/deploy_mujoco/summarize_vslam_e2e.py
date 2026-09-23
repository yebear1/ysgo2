#!/usr/bin/env python3
"""Merge mapping and localization phase metrics into one VSLAM report."""

import argparse
import json
import math
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    mapping = json.loads(args.mapping.read_text())
    localization = (
        json.loads(args.localization.read_text())
        if args.localization.is_file()
        else {
            "finished": False,
            "failed_reason": "skipped because mapping failed",
            "goals": [],
            "loop_closures": 0,
            "contact_episodes": 0,
            "tracking_lost_events": 0,
            "recovery_time_s": None,
            "truth_final_pose": None,
            "max_position_drift_m": None,
        }
    )
    thresholds = config["thresholds"]
    expected_goals = [item["name"] for item in config["localization"]["goals"]]
    reached_goals = [item["name"] for item in localization["goals"]]
    return_goal = next(
        (item for item in localization["goals"] if item["name"] == "start"),
        None,
    )
    return_position_error = (
        float(return_goal["truth_position_error_m"])
        if return_goal is not None else math.inf
    )
    truth_final = localization.get("truth_final_pose")
    desired_yaw = float(config["localization"]["goals"][-1].get("yaw", 0.0))
    return_yaw_error = (
        abs(math.atan2(
            math.sin(float(truth_final[2]) - desired_yaw),
            math.cos(float(truth_final[2]) - desired_yaw),
        )) if truth_final is not None else math.inf
    )
    recovery = localization.get("recovery_time_s")
    drift_values = [
        value for value in (
            mapping.get("max_position_drift_m"),
            localization.get("max_position_drift_m"),
        ) if value is not None
    ]
    max_position_drift = max(drift_values) if drift_values else None
    loops = int(mapping["loop_closures"]) + int(localization["loop_closures"])
    contacts = int(mapping["contact_episodes"]) + int(
        localization["contact_episodes"]
    )
    checks = {
        "mapping_finished": bool(mapping["finished"] and not mapping["failed_reason"]),
        "database_saved": args.database.is_file() and args.database.stat().st_size > 0,
        "localization_finished": bool(
            localization["finished"] and not localization["failed_reason"]
        ),
        "all_goals_reached": reached_goals == expected_goals,
        "zero_furniture_collisions": contacts <= int(
            thresholds["max_contact_episodes"]
        ),
        "return_position": return_position_error <= float(
            thresholds["return_position_error"]
        ),
        "return_yaw": math.degrees(return_yaw_error) <= float(
            thresholds["return_yaw_error_deg"]
        ),
        "tracking_loss_observed": localization["tracking_lost_events"] >= 1,
        "relocalized": recovery is not None,
        "recovery_time": recovery is not None and recovery <= float(
            thresholds["recovery_time"]
        ),
        "loop_closure": loops >= int(thresholds["minimum_loop_closures"]),
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "database": str(args.database),
        "database_size_bytes": (
            args.database.stat().st_size if args.database.is_file() else 0
        ),
        "expected_goals": expected_goals,
        "reached_goals": reached_goals,
        "total_contact_episodes": contacts,
        "total_loop_closures": loops,
        "return_position_error_m": (
            return_position_error if math.isfinite(return_position_error) else None
        ),
        "return_yaw_error_deg": (
            math.degrees(return_yaw_error)
            if math.isfinite(return_yaw_error) else None
        ),
        "max_position_drift_m": max_position_drift,
        "recovery_time_s": recovery,
        "mapping": mapping,
        "localization": localization,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Go2 VSLAM 全链路导航回归报告",
        "",
        f"- 总结果：{'通过' if report['passed'] else '失败'}",
        f"- 目标：{len(reached_goals)}/{len(expected_goals)}",
        f"- 家具碰撞：{contacts}",
        f"- 回环：{loops}",
        f"- 建图进度：{mapping.get('mapping_waypoints_reached', 0)}/"
        f"{mapping.get('mapping_waypoints_total', 0)}",
        f"- 建图失败原因：{mapping.get('failed_reason') or '无'}",
        f"- 定位失败原因：{localization.get('failed_reason') or '无'}",
        f"- 回到起点误差："
        f"{f'{return_position_error:.3f} m' if math.isfinite(return_position_error) else '不可用'}",
        f"- 最终航向误差："
        f"{f'{math.degrees(return_yaw_error):.2f}°' if math.isfinite(return_yaw_error) else '不可用'}",
        f"- 遮挡后重定位："
        f"{f'{recovery:.3f} s' if recovery is not None else '未恢复'}",
        f"- 全程最大定位漂移："
        f"{f'{max_position_drift:.3f} m' if max_position_drift is not None else '不可用'}",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        lines.extend(["", "未通过：" + "、".join(failed)])
    lines.extend(
        [
            "",
            "导航输入仅来自 RTAB-Map 位姿和保存的占据栅格。MuJoCo 真值只用于本报告评分。",
            "",
        ]
    )
    args.markdown_output.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
