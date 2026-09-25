#!/usr/bin/env python3
"""Sequential, failure-inclusive regression of a frozen saved-map controller."""

import argparse
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baselines/vslam-pass-20260925"
BASELINE_MAP_SHA = "4d9a47a4a9f584350202fa89b1200f91b1a65bf3869afa522079b368fe082872"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint():
    """Hash tracked executable inputs, including dirty contents, not reports."""
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    selected = [p for p in files if p and (
        p.startswith(("deploy/", "resources/robots/go2/", "legged_gym/"))
        or p in ("start_go2_vslam.sh", "run_vslam_e2e_benchmark.sh")
    )]
    return {p: digest(ROOT / p) for p in sorted(selected) if (ROOT / p).is_file()}


def cases():
    names = ("living_room", "bedroom", "kitchen")
    result = [{"id": f"order_{i + 1:02d}", "order": list(order), "initial_yaw_deg": 0.0}
              for i, order in enumerate(itertools.permutations(names))]
    result.extend({"id": f"heading_{i + 1:02d}", "order": list(names), "initial_yaw_deg": yaw}
                  for i, yaw in enumerate((-30.0, 30.0, 60.0, 180.0)))
    return result


def stall_episodes(trace, minimum_seconds=10.0, translation=0.15, rotation=math.radians(20)):
    """Scoring only: stationary despite active navigation, excluding lost vision.

    Translation or purposeful turning releases an episode; long episodes are
    counted once, not once per rolling window. Ground truth never feeds control.
    """
    events, anchor, event = [], None, None
    last_time = None

    def close(end):
        nonlocal event
        if event is not None:
            event["end_s"] = end
            event["duration_s"] = end - event["start_s"]
            events.append(event)
            event = None

    for row in trace:
        now = float(row["simulation_time"])
        control = row.get("control") or {}
        active = control.get("navigation_active", False) and not row.get("camera_blocked", False)
        pose = row.get("truth_pose")
        if not active or pose is None or row.get("vslam_pose") is None:
            close(last_time if last_time is not None else now)
            anchor = None
        else:
            moved = anchor is None
            if anchor is not None:
                old = anchor["truth_pose"]
                angle = math.atan2(math.sin(pose[2] - old[2]), math.cos(pose[2] - old[2]))
                moved = (math.dist(pose[:2], old[:2]) >= translation or abs(angle) >= rotation
                         or row.get("active_goal_index") != anchor.get("active_goal_index"))
            if moved:
                close(now)
                anchor = row
            elif event is None and now - anchor["simulation_time"] >= minimum_seconds:
                event = {"start_s": anchor["simulation_time"], "goal_index": row.get("active_goal_index")}
        last_time = now
    if last_time is not None:
        close(last_time)
    return events


def timing_stats(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return None
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": values[0], "max": values[-1],
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1]}


def aggregate(rows, planned_count):
    complete = [r for r in rows if r["status"] == "completed"]
    successes = sum(bool(r.get("passed")) for r in complete)
    failures = {}
    for row in complete:
        if not row.get("passed"):
            reason = row.get("failure_reason") or ", ".join(row.get("failed_checks", [])) or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
    return {
        "planned_runs": planned_count, "completed_runs": len(complete),
        "passed_runs": successes, "failed_runs": len(complete) - successes,
        "success_rate_all_planned": successes / planned_count,
        "total_known_contact_episodes": sum(r.get("contact_episodes") or 0 for r in complete),
        "runs_with_contacts": sum((r.get("contact_episodes") or 0) > 0 for r in complete),
        "runs_with_unknown_contact_metrics": sum(r.get("contact_episodes") is None for r in complete),
        "total_stall_episodes": sum(r.get("stall_episodes") or 0 for r in complete),
        "runs_with_stalls": sum((r.get("stall_episodes") or 0) > 0 for r in complete),
        "runs_with_unknown_stall_metrics": sum(r.get("stall_episodes") is None for r in complete),
        "wall_seconds_all_completed": timing_stats([r.get("wall_seconds") for r in complete]),
        "simulation_seconds_all_completed": timing_stats([r.get("simulation_seconds") for r in complete]),
        "failure_reasons": failures,
    }


def save_summary(output, manifest, rows):
    summary = aggregate(rows, len(manifest["cases"]))
    payload = {"schema_version": 1, "final": summary["completed_runs"] == summary["planned_runs"],
               "manifest": manifest, "summary": summary, "runs": rows}
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    lines = ["# VSLAM 批量回归", "",
             f"状态：{'完成' if payload['final'] else '进行中'}；已完成 {summary['completed_runs']}/{summary['planned_runs']}。",
             f"成功：{summary['passed_runs']}/{summary['planned_runs']}（所有预定样本作分母，未完成不算成功）。",
             f"已知接触事件：{summary['total_known_contact_episodes']}；卡住事件：{summary['total_stall_episodes']}。",
             f"缺少接触/卡住指标的运行：{summary['runs_with_unknown_contact_metrics']}/{summary['runs_with_unknown_stall_metrics']}。",
             "", "卡住定义：导航激活且视觉有效时，至少 10 仿真秒位移不足 0.15 m、航向变化不足 20°；遮挡/失定位时间单独排除。",
             "有意转身不计为卡住；长时间低效绕行仍可能超时，但不一定符合此静止判据。", "",
             "| 运行 | 顺序（最后均回起点） | 出发朝向 | 结果 | 碰撞 | 卡住 | 仿真秒 | 墙钟秒 | 原因 |",
             "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        def metric(key):
            value = row.get(key)
            return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
        result = "PASS" if row.get("passed") else "FAIL" if row["status"] == "completed" else row["status"]
        reason = str(row.get("failure_reason") or ", ".join(row.get("failed_checks", []))).replace("|", "/")
        lines.append(f"| {row['id']} | {' → '.join(row['order'])} | {row['initial_yaw_deg']:g}° | {result} | "
                     f"{metric('contact_episodes')} | {metric('stall_episodes')} | {metric('simulation_seconds')} | {metric('wall_seconds')} | {reason} |")
    for key, label in (("simulation_seconds_all_completed", "仿真耗时"), ("wall_seconds_all_completed", "墙钟耗时")):
        stats = summary[key]
        if stats:
            lines.extend(["", f"{label}（包含失败）：均值 {stats['mean']:.2f} s，中位数 {stats['median']:.2f} s，"
                          f"P95 {stats['p95_nearest_rank']:.2f} s，最大 {stats['max']:.2f} s。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def preflight_yaws(yaws):
    """Reject colliding test fixtures; never move furniture or robot XY to fit."""
    import mujoco
    import numpy as np
    config = yaml.safe_load((BASELINE / "go2_home.yaml").read_text())
    model = mujoco.MjModel.from_xml_path(str(ROOT / "resources/robots/go2/home_codex.xml"))
    for yaw in yaws:
        for standing in (False, True):
            data = mujoco.MjData(model)
            angle = math.radians(yaw)
            data.qpos[3:7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
            if standing:
                data.qpos[7:] = config["default_angles"]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
                if (b1 == 0) != (b2 == 0):
                    geom = contact.geom1 if b1 == 0 else contact.geom2
                    name = model.geom(geom).name
                    if name not in ("floor", "living_rug") and contact.dist < -1e-6:
                        raise RuntimeError(f"invalid fixture: yaw={yaw}, standing={standing}, overlap={name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--wall-timeout", type=float, default=900)
    parser.add_argument("--ros-domain", type=int, default=74)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = open(Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"go2_vslam_batch_{os.getuid()}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    matrix = cases()
    preflight_yaws(sorted({r["initial_yaw_deg"] for r in matrix}))
    database = output / "saved_map.rtabmap.db"
    with gzip.open(BASELINE / "map.rtabmap.db.gz", "rb") as src, database.open("wb") as dest:
        shutil.copyfileobj(src, dest)
    assert digest(database) == BASELINE_MAP_SHA, "baseline map checksum mismatch"
    shutil.copy2(BASELINE / "mapping.json", output / "mapping.json")
    source = source_fingerprint()
    manifest = {"baseline_tag": "vslam-pass-20260925", "baseline_commit": subprocess.check_output(
                    ["git", "rev-parse", "vslam-pass-20260925^{}"], cwd=ROOT, text=True).strip(),
                "test_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": source, "map_sha256": BASELINE_MAP_SHA, "cases": matrix,
                "wall_timeout_seconds": args.wall_timeout, "ros_domain": args.ros_domain,
                "policy": "No retries or parameter changes during the batch; all attempts retained. Final requested yaw is always 0 degrees.",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = [dict(case, status="pending") for case in matrix]
    config_template = yaml.safe_load((BASELINE / "benchmark.yaml").read_text())
    by_name = {item["name"]: item for item in config_template["localization"]["goals"]}
    for row in rows:
        if source_fingerprint() != source:
            raise RuntimeError("Source changed during batch: stop rather than mix controller versions")
        if digest(database) != BASELINE_MAP_SHA:
            raise RuntimeError("Saved map changed: stop rather than contaminate later trials")
        trial = output / row["id"]
        trial.mkdir()
        config = yaml.safe_load(yaml.safe_dump(config_template))
        config["localization"]["goals"] = [by_name[name] for name in row["order"] + ["start"]]
        (trial / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        row["status"] = "running"
        save_summary(output, manifest, rows)
        print(f"START {row['id']} yaw={row['initial_yaw_deg']} order={row['order']}", flush=True)
        env = dict(os.environ, ROS_DOMAIN_ID=str(args.ros_domain), PYTHONUNBUFFERED="1",
                   GO2_RTABMAP_LOG=str(trial / "rtabmap.log"))
        command = [str(ROOT / "start_go2_vslam.sh"), "--localization", "--no-rviz", f"--database={database}",
                   "--headless", "--vslam-benchmark-phase", "localization", "--vslam-benchmark-config", str(trial / "config.yaml"),
                   "--vslam-benchmark-output", str(trial / "localization.json"), "--vslam-initial-yaw-deg", str(row["initial_yaw_deg"])]
        started = time.monotonic()
        timed_out = False
        with (trial / "console.log").open("w") as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=args.wall_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        row.update(status="completed", wall_seconds=time.monotonic() - started, exit_code=process.returncode,
                   wall_timeout=timed_out, passed=False, contact_episodes=None, stall_episodes=None)
        row["map_unchanged"] = digest(database) == BASELINE_MAP_SHA
        result_path = trial / "localization.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            row.update(simulation_seconds=result.get("elapsed_simulation_s"), contact_episodes=result.get("contact_episodes"),
                       goals_reached=len(result.get("goals", [])), failure_reason=result.get("failed_reason"))
            trace_path = trial / "localization.trace.json"
            if trace_path.exists():
                stalls = stall_episodes(json.loads(trace_path.read_text()))
                row["stall_episodes"] = len(stalls)
                (trial / "stalls.json").write_text(json.dumps(stalls, indent=2) + "\n")
            summary_command = [sys.executable, str(ROOT / "deploy/deploy_mujoco/summarize_vslam_e2e.py"),
                               "--config", str(trial / "config.yaml"), "--mapping", str(output / "mapping.json"),
                               "--localization", str(result_path), "--database", str(database),
                               "--json-output", str(trial / "report.json"), "--markdown-output", str(trial / "report.md")]
            with (trial / "summary.log").open("w") as log:
                subprocess.run(summary_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            if (trial / "report.json").exists():
                report = json.loads((trial / "report.json").read_text())
                row["failed_checks"] = [key for key, passed in report["checks"].items() if not passed]
                row["passed"] = bool(report["passed"] and process.returncode == 0 and not timed_out and row["map_unchanged"])
            else:
                row["failure_reason"] = "report generation failed"
        else:
            row["failure_reason"] = "wall timeout without phase report" if timed_out else "process exited without phase report"
        if timed_out:
            row["failure_reason"] = "wall timeout"
        if not row["map_unchanged"]:
            row["failure_reason"] = "saved map checksum changed"
            row["passed"] = False
        (trial / "outcome.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        save_summary(output, manifest, rows)
        print(f"END {row['id']} passed={row['passed']} contacts={row['contact_episodes']} stalls={row['stall_episodes']} reason={row.get('failure_reason')}", flush=True)
    final = save_summary(output, manifest, rows)
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if final["summary"]["passed_runs"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
