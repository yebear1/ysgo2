"""State machine and truth-only scorer for the two-phase VSLAM benchmark."""

import json
import math
from pathlib import Path

import numpy as np
import yaml


def angle_difference(angle, reference):
    return math.atan2(math.sin(angle - reference), math.cos(angle - reference))


def yaw_from_quaternion(q):
    qw, qx, qy, qz = q
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


class VslamE2EBenchmark:
    """Drive with VSLAM products and keep simulator truth in the scorer only."""

    def __init__(self, config_path, phase, output_path):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.phase = str(phase)
        if self.phase not in ("mapping", "localization"):
            raise ValueError("benchmark phase must be mapping or localization")
        self.output_path = Path(output_path)
        self.phase_config = self.config[self.phase]
        self.timeout = float(self.phase_config["timeout"])
        self.finished = False
        self.failed_reason = None
        self.start_time = None
        self.last_time = None
        self.initial_truth = None
        self.initial_vslam = None
        self.last_truth = None
        self.last_vslam = None
        self.pose_errors = []
        self.yaw_errors = []
        self.contact_episodes = 0
        self.contact_events = []
        self.max_contact_force = 0.0
        self.strongest_contact = None
        self._contact_active = False
        self.map_updates = 0
        self.loop_closures = 0
        self.tracking_lost_events = 0
        self.longest_tracking_loss = 0.0
        self._lost_since = None
        self._last_pose_ready = False
        self.direct_target = None
        self.direct_yaw = None
        self._last_goal_reached = False
        self.goal_results = []
        self.camera_blocked = False
        self.occlusion_started = None
        self.occlusion_ended = None
        self.occlusion_lost_at = None
        self.relocalized_at = None
        self._occlusion_done = False
        self._waiting_relocalization = False
        self._pending_goal_index = 0
        self._mapping_index = 0
        self._mapping_progress_index = None
        self._mapping_best_distance = math.inf
        self._mapping_last_progress_time = None
        self._waypoint_changed = False
        self._final_heading_started = False
        self._blocked_since = None
        self._recover_until = None
        self.recovery_actions = 0
        self._pose_missing_since = None
        self._initial_pose_seen = False
        self.tracking_scan_actions = 0
        self._tracking_scan_active = False
        self._last_tracking_yaw = None
        self._last_diagnostic_time = None
        self._last_trace_time = -math.inf
        self.trace = []
        self._mapping_route_attempt_time = -math.inf

    def _elapsed(self, simulation_time):
        self.last_time = float(simulation_time)
        if self.start_time is None:
            self.start_time = float(simulation_time)
        return float(simulation_time) - self.start_time

    def _record_goal(self, goal_navigator):
        goal = np.asarray(goal_navigator.goal, dtype=np.float64)[:2]
        truth_error = (
            float(np.linalg.norm(self.last_truth[:2] - goal))
            if self.last_truth is not None
            else math.inf
        )
        self.goal_results.append(
            {
                "name": goal_navigator.goal_name,
                "goal": goal.tolist(),
                "truth_position_error_m": truth_error,
            }
        )

    def update_control(self, global_pose, goal_navigator, simulation_time):
        elapsed = self._elapsed(simulation_time)
        if elapsed > self.timeout:
            self.failed_reason = "timeout"
            self.finished = True
            return
        if self.camera_blocked:
            duration = float(self.phase_config.get("occlusion_duration", 2.5))
            if float(simulation_time) - self.occlusion_started >= duration:
                self.camera_blocked = False
                self.occlusion_ended = float(simulation_time)
            self.direct_target = None
            self.direct_yaw = None
            return
        if self.finished or global_pose is None:
            self.direct_target = None
            self.direct_yaw = None
            return

        if self.phase == "mapping":
            waypoints = self.phase_config["waypoints"]
            tolerance = float(self.phase_config.get("waypoint_tolerance", 0.40))
            while self._mapping_index < len(waypoints):
                target = np.asarray(waypoints[self._mapping_index], dtype=np.float64)
                distance = float(np.linalg.norm(target - global_pose[:2]))
                if self._mapping_progress_index != self._mapping_index:
                    self._mapping_progress_index = self._mapping_index
                    self._mapping_best_distance = distance
                    self._mapping_last_progress_time = float(simulation_time)
                progress_step = float(
                    self.phase_config.get("waypoint_progress_step", 0.15)
                )
                if distance <= self._mapping_best_distance - progress_step:
                    self._mapping_best_distance = distance
                    self._mapping_last_progress_time = float(simulation_time)
                progress_timeout = self.phase_config.get(
                    "waypoint_progress_timeout"
                )
                if (
                    progress_timeout is not None
                    and self._mapping_last_progress_time is not None
                    and float(simulation_time) - self._mapping_last_progress_time
                    > float(progress_timeout)
                ):
                    self.failed_reason = (
                        "mapping waypoint progress timeout: "
                        f"{self._mapping_index + 1}/{len(waypoints)}, "
                        f"best distance {self._mapping_best_distance:.3f} m"
                    )
                    self.direct_target = None
                    self.finished = True
                    return
                if distance > tolerance:
                    self.direct_target = self._exploration_target(
                        target, global_pose, goal_navigator, simulation_time
                    )
                    return
                print(
                    f"VSLAM mapping waypoint {self._mapping_index + 1}/"
                    f"{len(waypoints)} reached"
                )
                self._mapping_index += 1
                if goal_navigator is not None:
                    goal_navigator.cancel()
                self._mapping_progress_index = None
                self._mapping_best_distance = math.inf
                self._mapping_last_progress_time = None
                self._waypoint_changed = True
            self.direct_target = None
            self.finished = True
            return

        # A TF tree may be available before the saved grid has arrived or
        # before RTAB-Map has matched a frame against the saved database.
        # Neither state is yet sufficient to plan in the persistent map.
        if not goal_navigator.map_ready or self.loop_closures == 0:
            self.direct_target = None
            if elapsed > float(self.phase_config.get("initial_tracking_timeout", 20.0)):
                self.failed_reason = "saved-map localization or occupancy grid unavailable"
                self.finished = True
            return

        goals = self.phase_config["goals"]
        if goal_navigator.reached and not self._last_goal_reached:
            self._record_goal(goal_navigator)
            reached_name = goal_navigator.goal_name
            if (
                reached_name == self.phase_config.get("occlusion_after_goal")
                and not self._occlusion_done
            ):
                self.camera_blocked = True
                self.occlusion_started = float(simulation_time)
                self._waiting_relocalization = True
            self._pending_goal_index += 1
        self._last_goal_reached = bool(goal_navigator.reached)

        if self._waiting_relocalization:
            if self.occlusion_lost_at is not None and global_pose is not None:
                self.relocalized_at = float(simulation_time)
                self._waiting_relocalization = False
                self._occlusion_done = True
            else:
                return

        if self._pending_goal_index >= len(goals):
            final_yaw = goals[-1].get("yaw")
            if final_yaw is not None:
                error = angle_difference(float(final_yaw), float(global_pose[2]))
                if abs(error) > math.radians(4.0):
                    self.direct_yaw = float(final_yaw)
                    self._final_heading_started = True
                    return
            self.direct_yaw = None
            self.finished = True
            return

        if not goal_navigator.active and not self._last_goal_reached:
            item = goals[self._pending_goal_index]
            success = goal_navigator.set_goal(
                item["position"], global_pose[:2], simulation_time, item["name"]
            )
            if not success:
                self.failed_reason = f"no path to {item['name']}"
                self.finished = True
        elif self._last_goal_reached:
            goal_navigator.cancel()
            self._last_goal_reached = False

    def _exploration_target(self, target, pose, navigator, simulation_time):
        """Use observed free-space routes when returning through mapped rooms.

        Unobserved exploration goals still use the local sensor controller.
        Once the map contains a goal, a straight ray through furniture is not
        a valid return route: follow the existing grid planner's waypoints.
        """
        if navigator is None or not navigator.map_ready:
            return target
        if navigator.active:
            navigator.update(pose[:2], pose[2], simulation_time)
            if navigator.active:
                return navigator.path[min(navigator.waypoint_index, len(navigator.path) - 1)]
            return target
        if float(simulation_time) - self._mapping_route_attempt_time < 1.0:
            return target
        if not navigator._world_in_bounds(target) or not navigator._world_in_bounds(pose[:2]):
            return target
        start_cell = navigator._world_to_grid(pose[:2])
        goal_cell = navigator._world_to_grid(target)
        if not navigator.known[goal_cell] or navigator.occupied[goal_cell]:
            return target
        if navigator._line_clear(start_cell, goal_cell):
            return target
        self._mapping_route_attempt_time = float(simulation_time)
        if navigator.set_goal(target, pose[:2], simulation_time, "mapping_return"):
            return navigator.path[min(navigator.waypoint_index, len(navigator.path) - 1)]
        return target

    def consume_waypoint_change(self):
        """Return a one-shot signal for resetting local reactive state."""
        changed = self._waypoint_changed
        self._waypoint_changed = False
        return changed

    def recovery_command(self, navigator_state, simulation_time):
        """Back out of a local dead end using only navigator state."""
        if self.phase != "mapping" or self.finished:
            return None
        now = float(simulation_time)
        if self._recover_until is not None:
            if now < self._recover_until:
                return np.array([-0.22, 0.0, 0.0], dtype=np.float32)
            self._recover_until = None
            self._blocked_since = None
            return None
        if navigator_state == "BLOCKED":
            if self._blocked_since is None:
                self._blocked_since = now
            elif now - self._blocked_since >= 1.0:
                self._recover_until = now + 1.5
                self.recovery_actions += 1
                print(f"VSLAM exploration recovery #{self.recovery_actions}")
                return np.array([-0.22, 0.0, 0.0], dtype=np.float32)
        else:
            self._blocked_since = None
        return None

    def tracking_recovery_command(self, global_pose, simulation_time, sensor_yaw=None, lidar=None):
        """Actively search for visual features after tracking is lost.

        Uses RTAB-Map validity, the clock, gyro heading and measured ranges.
        MuJoCo truth is never consulted. Alternating a slow
        yaw scan gives visual odometry new parallax while avoiding the blind,
        indefinite stop that a lost pose previously caused.
        """
        if self.finished:
            return None
        now = float(simulation_time)
        if global_pose is not None:
            if sensor_yaw is not None:
                self._last_tracking_yaw = float(sensor_yaw)
            self._initial_pose_seen = True
            self._pose_missing_since = None
            self._tracking_scan_active = False
            return None
        if self.camera_blocked:
            # Do not move while the deliberately blinded camera cannot provide
            # any feedback.  Scanning starts as soon as the cover is removed.
            return np.zeros(3, dtype=np.float32)
        if self._pose_missing_since is None:
            self._pose_missing_since = now
        missing_for = now - self._pose_missing_since
        initial_grace = float(self.phase_config.get("initial_tracking_grace", 8.0))
        scan_delay = float(self.phase_config.get("tracking_scan_delay", 0.5))
        if not self._initial_pose_seen and missing_for < initial_grace:
            return np.zeros(3, dtype=np.float32)
        timeout_key = (
            "tracking_recovery_timeout"
            if self._initial_pose_seen
            else "initial_tracking_timeout"
        )
        timeout = float(self.phase_config.get(timeout_key, 12.0))
        if missing_for > timeout:
            self.failed_reason = "visual tracking recovery timeout"
            self.finished = True
            return np.zeros(3, dtype=np.float32)
        if missing_for < scan_delay:
            return np.zeros(3, dtype=np.float32)
        if lidar is not None and missing_for < 2.0:
            # A nearby featureless face fills the image; yaw recovery can
            # trigger endless sideways turn-clearance escapes. First create
            # camera stand-off, only if measured rear swept space is free.
            near_face = lidar.planar_clearance(0.0, math.radians(20.0)) < 0.65
            rear_space = lidar.translation_clearance(
                math.pi, half_width=0.20, body_half_length=0.39
            )
            if near_face and rear_space > 0.20:
                return np.array([-0.15, 0.0, 0.0], dtype=np.float32)
        if not self._tracking_scan_active:
            self._tracking_scan_active = True
            self.tracking_scan_actions += 1
            print(f"VSLAM active relocalization scan #{self.tracking_scan_actions}")
            max_scans = self.phase_config.get("max_tracking_loss_events")
            if (
                max_scans is not None
                and self.tracking_scan_actions > int(max_scans)
            ):
                self.failed_reason = (
                    "repeated visual tracking loss: "
                    f"{self.tracking_scan_actions} events"
                )
                self.finished = True
                return np.zeros(3, dtype=np.float32)
        period = float(self.phase_config.get("tracking_scan_period", 2.0))
        direction = 1.0 if int(missing_for / period) % 2 == 0 else -1.0
        yaw_rate = float(self.phase_config.get("tracking_scan_yaw_rate", 0.35))
        if self._last_tracking_yaw is not None and sensor_yaw is not None:
            # Return to the last observed view first. An open-loop alternating
            # velocity can keep looking away from it, especially when obstacle
            # avoidance interrupts one half of the turn.
            phase = max(0, int((missing_for - scan_delay) / period))
            offsets = (0.0, -0.30, 0.30, -0.60, 0.60)
            target = self._last_tracking_yaw + offsets[min(phase, len(offsets) - 1)]
            direction = np.clip(
                1.8 * angle_difference(target, float(sensor_yaw)),
                -yaw_rate, yaw_rate,
            )
            return np.array([0.0, 0.0, direction], dtype=np.float32)
        return np.array([0.0, 0.0, direction * yaw_rate], dtype=np.float32)

    def heading_command(self, global_pose, max_yaw_rate=0.9):
        if self.direct_yaw is None or global_pose is None:
            return None
        error = angle_difference(self.direct_yaw, float(global_pose[2]))
        return np.array(
            [0.0, 0.0, np.clip(1.8 * error, -max_yaw_rate, max_yaw_rate)],
            dtype=np.float32,
        )

    def observe(
        self,
        simulation_time,
        truth_pose,
        global_pose,
        contact,
        bridge,
        control_diagnostics=None,
    ):
        self._elapsed(simulation_time)
        truth_pose = np.asarray(truth_pose, dtype=np.float64)
        self.last_truth = truth_pose.copy()
        self.last_vslam = None if global_pose is None else np.asarray(global_pose).copy()
        self.map_updates = int(bridge.map_updates)
        self.loop_closures = int(bridge.loop_closures)

        pose_ready = global_pose is not None
        if self._last_pose_ready and not pose_ready:
            self.tracking_lost_events += 1
            self._lost_since = float(simulation_time)
            if self._waiting_relocalization and self.occlusion_lost_at is None:
                self.occlusion_lost_at = float(simulation_time)
        if pose_ready and self._lost_since is not None:
            duration = float(simulation_time) - self._lost_since
            self.longest_tracking_loss = max(self.longest_tracking_loss, duration)
            self._lost_since = None
        self._last_pose_ready = pose_ready

        if float(simulation_time) - self._last_trace_time >= 0.25:
            self._last_trace_time = float(simulation_time)
            self.trace.append({
                "simulation_time": float(simulation_time),
                "waypoints_reached": self._mapping_index,
                "truth_pose": truth_pose.tolist(),
                "vslam_pose": None if self.last_vslam is None else self.last_vslam.tolist(),
                "inliers": getattr(bridge, "odom_inliers", None),
                "matches": getattr(bridge, "odom_matches", None),
                "features": getattr(bridge, "odom_features", None),
                "tracking_mode": bridge.tracking_mode,
                "camera_frames": getattr(bridge, "camera_frames", None),
                "odometry_messages": getattr(bridge, "odometry_messages", None),
                "odometry_age_s": getattr(bridge, "odometry_age", None),
                "control": control_diagnostics,
            })

        if global_pose is not None:
            global_pose = np.asarray(global_pose, dtype=np.float64)
            if self.initial_truth is None:
                self.initial_truth = truth_pose.copy()
                self.initial_vslam = global_pose.copy()
            yaw_offset = self.initial_truth[2] - self.initial_vslam[2]
            c, s = math.cos(yaw_offset), math.sin(yaw_offset)
            rotation = np.array([[c, -s], [s, c]])
            predicted = self.initial_truth[:2] + rotation @ (
                global_pose[:2] - self.initial_vslam[:2]
            )
            self.pose_errors.append(float(np.linalg.norm(predicted - truth_pose[:2])))
            predicted_yaw = global_pose[2] + yaw_offset
            self.yaw_errors.append(abs(angle_difference(predicted_yaw, truth_pose[2])))
            if (
                self._last_diagnostic_time is None
                or float(simulation_time) - self._last_diagnostic_time >= 10.0
            ):
                self._last_diagnostic_time = float(simulation_time)
                print(
                    "VSLAM score-only diagnostic: "
                    f"position_drift={self.pose_errors[-1]:.3f}m, "
                    f"yaw_drift={math.degrees(self.yaw_errors[-1]):.2f}deg"
                )
            abort_drift = self.phase_config.get("abort_position_drift")
            if (
                abort_drift is not None
                and self._elapsed(simulation_time) >= 10.0
                and self.pose_errors[-1] > float(abort_drift)
            ):
                self.failed_reason = (
                    "position drift exceeded scoring bound: "
                    f"{self.pose_errors[-1]:.3f} m"
                )
                self.finished = True

        force = 0.0 if contact is None else float(contact[0])
        active = force >= 5.0
        if active and not self._contact_active:
            self.contact_episodes += 1
            event = {
                "simulation_time": float(simulation_time),
                "force_n": force,
                "geoms": [contact[1], contact[2]],
                "truth_pose": truth_pose.tolist(),
                "control": control_diagnostics,
            }
            self.contact_events.append(event)
            print("VSLAM score-only contact: " + json.dumps(event), flush=True)
        self._contact_active = active
        if force > self.max_contact_force:
            self.max_contact_force = force
            self.strongest_contact = None if contact is None else [contact[1], contact[2]]

    def result(self, bridge):
        recovery = None
        if self.occlusion_ended is not None and self.relocalized_at is not None:
            recovery = max(0.0, self.relocalized_at - self.occlusion_ended)
        longest_tracking_loss = self.longest_tracking_loss
        if self._lost_since is not None and self.last_time is not None:
            longest_tracking_loss = max(
                longest_tracking_loss,
                float(self.last_time) - self._lost_since,
            )
        result = {
            "schema_version": 1,
            "phase": self.phase,
            "finished": self.finished,
            "failed_reason": self.failed_reason,
            "elapsed_simulation_s": (
                0.0 if self.last_time is None or self.start_time is None
                else self.last_time - self.start_time
            ),
            "map_updates": self.map_updates,
            "loop_closures": self.loop_closures,
            "tracking_lost_events": self.tracking_lost_events,
            "longest_tracking_loss_s": longest_tracking_loss,
            "recovery_time_s": recovery,
            "contact_episodes": self.contact_episodes,
            "contact_events": self.contact_events,
            "max_contact_force_n": self.max_contact_force,
            "strongest_contact": self.strongest_contact,
            "mean_position_drift_m": (
                float(np.mean(self.pose_errors)) if self.pose_errors else None
            ),
            "max_position_drift_m": (
                float(np.max(self.pose_errors)) if self.pose_errors else None
            ),
            "mean_yaw_drift_deg": (
                math.degrees(float(np.mean(self.yaw_errors)))
                if self.yaw_errors else None
            ),
            "max_yaw_drift_deg": (
                math.degrees(float(np.max(self.yaw_errors)))
                if self.yaw_errors else None
            ),
            "goals": self.goal_results,
            "truth_final_pose": (
                self.last_truth.tolist() if self.last_truth is not None else None
            ),
            "vslam_final_pose": (
                self.last_vslam.tolist() if self.last_vslam is not None else None
            ),
            "tracking_mode": bridge.tracking_mode,
            "exploration_recoveries": self.recovery_actions,
            "tracking_scan_actions": self.tracking_scan_actions,
            "mapping_waypoints_reached": self._mapping_index,
            "mapping_waypoints_total": (
                len(self.phase_config.get("waypoints", []))
                if self.phase == "mapping"
                else None
            ),
        }
        return result

    def write_result(self, bridge):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.with_suffix(".trace.json").write_text(
            json.dumps(self.trace, ensure_ascii=False, indent=2) + "\n"
        )
        grid = getattr(bridge, "_latest_map", None)
        if grid is not None:
            np.savez_compressed(
                self.output_path.with_suffix(".map.npz"),
                data=grid["data"], origin=grid["origin"], resolution=grid["resolution"],
            )
        self.output_path.write_text(
            json.dumps(self.result(bridge), ensure_ascii=False, indent=2) + "\n"
        )
