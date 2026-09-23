"""Reactive traversability navigation over a local LiDAR elevation map."""

import math

import numpy as np


def _angle_difference(angle, reference):
    return math.atan2(math.sin(angle - reference), math.cos(angle - reference))


class TerrainNavigator:
    """Turn away from untraversable positive and negative height jumps.

    The navigator does not classify named terrain. It only decides whether a
    body-width corridor is traversable. Traversable geometry remains entirely
    under the terrain-aware RL policy's control.
    """

    def __init__(self, config):
        self.enabled = bool(config.get("enabled", True))
        self.max_step_up = float(config.get("max_step_up", 0.18))
        self.max_drop = float(config.get("max_drop", 0.15))
        self.lookahead_min = float(config.get("lookahead_min", 0.25))
        self.lookahead_max = float(config.get("lookahead_max", 1.00))
        self.planning_lookahead = float(config.get("planning_lookahead", 1.40))
        self.corridor_half_width = float(config.get("corridor_half_width", 0.22))
        self.planning_clearance = float(config.get("planning_clearance", 0.12))
        self.hazard_fraction = float(config.get("hazard_fraction", 0.45))
        self.drop_fraction = float(config.get("drop_fraction", 0.15))
        self.turn_speed = float(config.get("turn_speed", 0.90))
        self.heading_step = math.radians(float(config.get("heading_step_deg", 10.0)))
        self.max_search_angle = math.radians(
            float(config.get("max_search_angle_deg", 90.0))
        )
        self.heading_tolerance = math.radians(
            float(config.get("heading_tolerance_deg", 6.0))
        )
        self.clear_scans_required = int(config.get("clear_scans_required", 3))
        self.goal_obstacle_margin = float(config.get("goal_obstacle_margin", 0.05))
        self.turning_radius = float(config.get("turning_radius", 0.43))
        self.backup_speed = float(config.get("backup_speed", 0.25))
        self.rear_stop_distance = float(config.get("rear_stop_distance", 0.30))
        self.backup_updates_limit = int(config.get("backup_updates", 150))
        self.advance_speed = float(config.get("advance_speed", 0.22))
        self.forward_stop_distance = float(
            config.get("forward_stop_distance", 0.20)
        )
        self.advance_updates_limit = int(config.get("advance_updates", 100))
        self.body_half_length = float(config.get("body_half_length", 0.39))
        self.lateral_escape_speed = float(
            config.get("lateral_escape_speed", 0.20)
        )
        self.lateral_stop_distance = float(
            config.get("lateral_stop_distance", 0.12)
        )
        self.lateral_escape_updates_limit = int(
            config.get("lateral_escape_updates", 120)
        )
        self.forward_arc_max_angle = math.radians(
            float(config.get("forward_arc_max_deg", 15.0))
        )
        self.forward_arc_speed = float(config.get("forward_arc_speed", 0.35))
        self.forward_arc_switch_penalty = float(
            config.get("forward_arc_switch_penalty", 0.25)
        )
        self.centering_enabled = bool(config.get("centering_enabled", True))
        self.centering_activation = float(
            config.get("centering_activation", 0.75)
        )
        self.centering_target = float(config.get("centering_target", 0.30))
        self.centering_gain = float(config.get("centering_gain", 0.80))
        self.centering_velocity_gain = float(
            config.get("centering_velocity_gain", 0.50)
        )
        self.centering_velocity_deadband = float(
            config.get("centering_velocity_deadband", 0.03)
        )
        self.centering_max_lateral = float(
            config.get("centering_max_lateral", 0.20)
        )
        self.centering_deadband = float(config.get("centering_deadband", 0.025))
        self.centering_side_half_angle = math.radians(
            float(config.get("centering_side_half_angle_deg", 12.0))
        )
        self.centering_forward_speed = float(
            config.get("centering_forward_speed", 0.75)
        )
        self.centering_heading_gain = float(config.get("centering_heading_gain", 1.8))
        self.centering_narrow_width = float(config.get("centering_narrow_width", 0.8))
        self.centering_narrow_speed = float(config.get("centering_narrow_speed", 0.35))

        self.state = "FORWARD"
        self.hazard = "CLEAR"
        self.hazard_distance = math.inf
        self.turn_direction = 1.0
        self.last_turn_direction = -1.0
        self.turn_start_yaw = 0.0
        self.target_yaw = 0.0
        self.target_relative_angle = 0.0
        self.clear_scans = 0
        self.resume_speed = 0.0
        self.backup_updates = 0
        self.lateral_escape_direction = 0.0
        self.centering_lateral = 0.0

    def reset(self):
        self.state = "FORWARD"
        self.hazard = "CLEAR"
        self.hazard_distance = math.inf
        self.clear_scans = 0
        self.resume_speed = 0.0
        self.backup_updates = 0
        self.lateral_escape_direction = 0.0
        self.centering_lateral = 0.0

    def _translation_clearance(self, lidar, angle, lateral=False):
        """Measure travel outside the physical rectangular body envelope."""
        if lateral:
            return lidar.translation_clearance(
                angle,
                half_width=self.body_half_length,
                body_half_length=self.corridor_half_width,
            )
        return lidar.translation_clearance(
            angle,
            half_width=self.corridor_half_width,
            body_half_length=self.body_half_length,
        )

    def _choose_lateral_escape(self, lidar):
        """Move away from a close side wall before attempting a pivot."""
        left = self._translation_clearance(lidar, math.pi / 2.0, lateral=True)
        right = self._translation_clearance(lidar, -math.pi / 2.0, lateral=True)
        direction, clearance = (
            (1.0, left) if left >= right else (-1.0, right)
        )
        needed = max(
            self.lateral_stop_distance,
            self.turning_radius - lidar.planar_clearance() + 0.03,
        )
        return (direction, clearance) if clearance > needed else (0.0, clearance)

    def _center_forward_command(self, command, lidar, lateral_velocity=0.0):
        """Keep the body centred using side ranges and measured lateral drift."""
        result = np.asarray(command, dtype=np.float32).copy()
        self.centering_lateral = 0.0
        if (
            not self.centering_enabled
            or result[0] <= 1.0e-3
            or abs(result[1]) > 1.0e-3
            or abs(result[2]) > 0.35
        ):
            return result

        left = lidar.planar_clearance(
            math.pi / 2.0, self.centering_side_half_angle
        )
        right = lidar.planar_clearance(
            -math.pi / 2.0, self.centering_side_half_angle
        )
        left_near = left < self.centering_activation
        right_near = right < self.centering_activation

        if left_near and right_near:
            # Positive body Y is left.  Move toward the side with more room.
            error = 0.5 * (left - right)
            result[0] = min(float(result[0]), self.centering_forward_speed)
        elif left < self.centering_target:
            error = -(self.centering_target - left)
        elif right < self.centering_target:
            error = self.centering_target - right
        else:
            return result

        alignment = lidar.corridor_alignment(self.centering_activation)
        if alignment is not None:
            error, heading, width = alignment
            # Align the body with the walls as well as translating towards
            # their centre. Otherwise a forward command keeps driving across
            # the passage and fights the lateral correction.
            if self.state == "FORWARD":
                result[2] = np.clip(
                    self.centering_heading_gain * heading,
                    -self.turn_speed,
                    self.turn_speed,
                )
            clearance_ratio = np.clip(
                (width - self.centering_narrow_width)
                / max(2.0 * self.centering_activation - self.centering_narrow_width, 1.0e-3),
                0.0,
                1.0,
            )
            speed_limit = self.centering_narrow_speed + clearance_ratio * (
                self.centering_forward_speed - self.centering_narrow_speed
            )
            result[0] = min(float(result[0]), speed_limit)

        position_correction = (
            0.0 if abs(error) <= self.centering_deadband
            else self.centering_gain * error
        )
        measured_lateral = float(lateral_velocity)
        velocity_correction = (
            0.0
            if abs(measured_lateral) <= self.centering_velocity_deadband
            else -self.centering_velocity_gain * measured_lateral
        )
        if position_correction == 0.0 and velocity_correction == 0.0:
            return result
        lateral = float(
            np.clip(
                position_correction + velocity_correction,
                -self.centering_max_lateral,
                self.centering_max_lateral,
            )
        )
        result[1] = lateral
        self.centering_lateral = lateral
        return result

    @staticmethod
    def _corridor_hazard(
        elevation,
        x_values,
        y_values,
        y_mask,
        x_mask,
        max_step_up,
        max_drop,
        hazard_fraction,
        drop_fraction,
    ):
        x_indices = np.flatnonzero(x_mask)
        y_indices = np.flatnonzero(y_mask)
        if not x_indices.size or not y_indices.size:
            return "CLEAR", math.inf

        previous = np.zeros(y_indices.size, dtype=np.float32)
        previous_valid = np.ones(y_indices.size, dtype=bool)
        for x_index in x_indices:
            current = elevation[x_index, y_indices]
            valid = np.isfinite(current)
            delta = current - previous
            obstacle = valid & previous_valid & (delta > max_step_up)
            drop = (~valid & previous_valid) | (
                valid & previous_valid & (delta < -max_drop)
            )
            if np.mean(obstacle) >= hazard_fraction:
                return "OBSTACLE", float(x_values[x_index])
            # Losing even one side of a foot-width corridor is dangerous near
            # a platform corner, so drops deliberately use a lower threshold.
            if np.mean(drop) >= drop_fraction:
                return "DROP", float(x_values[x_index])
            previous = np.where(valid, current, previous)
            previous_valid = valid
        return "CLEAR", math.inf

    def analyze(self, lidar, side=0):
        x_mask = (lidar.x_values >= self.lookahead_min) & (
            lidar.x_values <= self.lookahead_max
        )
        if side > 0:
            y_mask = (lidar.y_values >= 0.12) & (
                lidar.y_values <= self.corridor_half_width + 0.35
            )
        elif side < 0:
            y_mask = (lidar.y_values <= -0.12) & (
                lidar.y_values >= -self.corridor_half_width - 0.35
            )
        else:
            y_mask = np.abs(lidar.y_values) <= self.corridor_half_width
        return self._corridor_hazard(
            lidar.elevation,
            lidar.x_values,
            lidar.y_values,
            y_mask,
            x_mask,
            self.max_step_up,
            self.max_drop,
            self.hazard_fraction,
            self.drop_fraction,
        )

    def _candidate_result(self, lidar, angle):
        x_values = lidar.x_values[
            (lidar.x_values >= self.lookahead_min)
            & (lidar.x_values <= self.planning_lookahead)
        ]
        y_values = lidar.y_values[
            np.abs(lidar.y_values)
            <= self.corridor_half_width + self.planning_clearance
        ]
        patch = lidar.sample_corridor(angle, x_values, y_values)
        if not patch.size:
            return "DROP", math.inf
        hazard, _ = self._corridor_hazard(
            patch,
            x_values,
            y_values,
            np.ones(len(y_values), dtype=bool),
            np.ones(len(x_values), dtype=bool),
            self.max_step_up,
            self.max_drop,
            self.hazard_fraction,
            self.drop_fraction,
        )
        missing = np.mean(~np.isfinite(patch))
        values = np.nan_to_num(patch, nan=0.0)
        jumps = np.diff(np.vstack((np.zeros((1, values.shape[1])), values)), axis=0)
        unsafe = np.mean((jumps > self.max_step_up) | (jumps < -self.max_drop))
        roughness = float(np.mean(np.abs(jumps)))
        cost = float(8.0 * missing + 5.0 * unsafe + roughness)
        return hazard, cost

    def _choose_heading(self, lidar):
        """Return the smallest safe heading change, not a fixed turn angle."""
        fallback = []
        steps = max(1, int(round(self.max_search_angle / self.heading_step)))
        for step in range(1, steps + 1):
            magnitude = min(step * self.heading_step, self.max_search_angle)
            safe = []
            for direction in (self.last_turn_direction, -self.last_turn_direction):
                angle = direction * magnitude
                hazard, cost = self._candidate_result(lidar, angle)
                fallback.append((cost, abs(angle), angle))
                if hazard == "CLEAR":
                    safe.append((cost, angle))
            if safe:
                _, angle = min(safe, key=lambda item: item[0])
                self.last_turn_direction = 1.0 if angle > 0.0 else -1.0
                return angle

        # If every sampled corridor is blocked, turn toward the least-bad one
        # and scan again from the new orientation.
        _, _, angle = min(fallback)
        self.last_turn_direction = 1.0 if angle > 0.0 else -1.0
        return angle

    def _set_target_heading(self, lidar, yaw):
        self.target_relative_angle = self._choose_heading(lidar)
        self.target_yaw = float(yaw) + self.target_relative_angle
        self.turn_direction = 1.0 if self.target_relative_angle > 0.0 else -1.0
        self.turn_start_yaw = float(yaw)
        self.clear_scans = 0

    def _choose_forward_arc(self, lidar):
        """Find a small traversable correction that does not require a pivot.

        Object geometry stays at its measured boundary.  This check sweeps the
        configured body-width corridor along a shallow forward arc, so a Go2
        that fits through a passage can centre itself without applying the
        much larger in-place turning envelope to nearby furniture.
        """
        steps = max(1, int(self.forward_arc_max_angle / self.heading_step + 1.0e-9))
        for step in range(1, steps + 1):
            magnitude = step * self.heading_step
            safe = []
            for direction in (self.last_turn_direction, -self.last_turn_direction):
                angle = direction * magnitude
                hazard, cost = self._candidate_result(lidar, angle)
                if hazard == "CLEAR":
                    switch_cost = (
                        self.forward_arc_switch_penalty
                        if direction != self.last_turn_direction
                        else 0.0
                    )
                    safe.append((cost + switch_cost, angle))
            if safe:
                _, angle = min(safe, key=lambda item: item[0])
                self.last_turn_direction = 1.0 if angle > 0.0 else -1.0
                return angle
        return None

    def _begin_forward_arc(
        self, angle, yaw, command, lidar, lateral_velocity=0.0
    ):
        self.state = "STEER_FORWARD"
        self.target_relative_angle = float(angle)
        self.target_yaw = float(yaw) + float(angle)
        self.turn_direction = 1.0 if angle > 0.0 else -1.0
        self.turn_start_yaw = float(yaw)
        self.resume_speed = float(command[0])
        self.clear_scans = 0
        yaw_rate = float(
            np.clip(1.8 * angle, -self.turn_speed, self.turn_speed)
        )
        return self._center_forward_command(
            np.array(
                [min(float(command[0]), self.forward_arc_speed), 0.0, yaw_rate],
                dtype=np.float32,
            ),
            lidar,
            lateral_velocity,
        )

    def update(
        self,
        manual_command,
        lidar,
        yaw,
        autonomous=False,
        goal_distance=None,
        lateral_velocity=0.0,
    ):
        command = np.asarray(manual_command, dtype=np.float32).copy()
        if not self.enabled or lidar is None:
            return command

        # Explicit stop, reverse and lateral commands override autonomy.
        if (not autonomous) and (
            command[0] < -1.0e-3
            or abs(command[1]) > 1.0e-3
            or (
                abs(command[0]) <= 1.0e-3
                and abs(command[2]) <= 1.0e-3
            )
        ):
            self.reset()
            return command

        # Continue an already selected recovery direction before interpreting
        # the still-present yaw command again.
        if self.state in (
            "LATERAL_FOR_TURN",
            "ADVANCE_FOR_TURN",
            "BACKUP_FOR_TURN",
        ):
            turn_clearance = lidar.planar_clearance()
            self.hazard = "TURN_SWEEP"
            self.hazard_distance = turn_clearance
            if turn_clearance >= self.turning_radius:
                self.reset()
                return command
            if self.state == "LATERAL_FOR_TURN":
                direction = self.lateral_escape_direction
                travel_clearance = self._translation_clearance(
                    lidar, direction * math.pi / 2.0, lateral=True
                )
                speed = direction * self.lateral_escape_speed
                limit = self.lateral_escape_updates_limit
            elif self.state == "ADVANCE_FOR_TURN":
                travel_clearance = self._translation_clearance(lidar, 0.0)
                speed = self.advance_speed
                limit = self.advance_updates_limit
            else:
                travel_clearance = self._translation_clearance(lidar, math.pi)
                speed = -self.backup_speed
                limit = self.backup_updates_limit
            if (
                travel_clearance
                > (
                    self.lateral_stop_distance
                    if self.state == "LATERAL_FOR_TURN"
                    else self.forward_stop_distance
                )
                and self.backup_updates < limit
            ):
                self.backup_updates += 1
                if self.state == "LATERAL_FOR_TURN":
                    return np.array([0.0, speed, 0.0], dtype=np.float32)
                return np.array([speed, 0.0, 0.0], dtype=np.float32)
            self.state = "BLOCKED"
            return np.zeros(3, dtype=np.float32)

        if self.state == "STEER_FORWARD":
            forward_hazard, forward_distance = self.analyze(lidar)
            heading_error = _angle_difference(self.target_yaw, float(yaw))
            self.hazard = forward_hazard
            self.hazard_distance = forward_distance
            if (
                forward_hazard == "CLEAR"
                and abs(heading_error) <= self.heading_tolerance
            ):
                self.reset()
                return self._center_forward_command(
                    command, lidar, lateral_velocity
                )
            if abs(heading_error) <= self.heading_tolerance:
                angle = self._choose_forward_arc(lidar)
                if angle is None:
                    self.state = "BLOCKED"
                    return np.zeros(3, dtype=np.float32)
                return self._begin_forward_arc(
                    angle, yaw, command, lidar, lateral_velocity
                )
            yaw_rate = float(
                np.clip(1.8 * heading_error, -self.turn_speed, self.turn_speed)
            )
            return self._center_forward_command(
                np.array(
                    [min(float(command[0]), self.forward_arc_speed), 0.0, yaw_rate],
                    dtype=np.float32,
                ),
                lidar,
                lateral_velocity,
            )

        if self.state == "BLOCKED":
            # BLOCKED means there was not enough room for the previously
            # requested in-place turn.  It must not latch a later straight
            # forward command when the body-width corridor ahead is clear.
            forward_hazard, forward_distance = self.analyze(lidar)
            # A learned controller almost always adds a small lateral/yaw
            # correction to a forward command.  Treat that as forward motion,
            # not as a new in-place-pivot request, otherwise BLOCKED can latch
            # forever even after the measured body-width corridor is clear.
            forward_motion = command[0] > 1.0e-3
            if forward_motion and forward_hazard == "CLEAR":
                self.reset()
                return command
            if forward_motion:
                angle = self._choose_forward_arc(lidar)
                if angle is not None:
                    return self._begin_forward_arc(
                        angle, yaw, command, lidar, lateral_velocity
                    )
            self.hazard = (
                "TURN_SWEEP" if forward_hazard == "CLEAR" else forward_hazard
            )
            self.hazard_distance = (
                lidar.planar_clearance()
                if forward_hazard == "CLEAR"
                else forward_distance
            )
            if lidar.planar_clearance() >= self.turning_radius:
                self.reset()
                return command
            return np.zeros(3, dtype=np.float32)

        # A point-foot path can look clear while a 0.72 m long quadruped clips
        # a wall with a rear thigh during an in-place turn. Protect both manual
        # and autonomous yaw commands with the full swept turning radius.
        if abs(command[2]) > 1.0e-3 and command[0] <= 0.0:
            turn_clearance = lidar.planar_clearance()
            if turn_clearance < self.turning_radius:
                lateral_direction, _ = self._choose_lateral_escape(lidar)
                forward_clearance = self._translation_clearance(lidar, 0.0)
                rear_clearance = self._translation_clearance(lidar, math.pi)
                self.hazard = "TURN_SWEEP"
                self.hazard_distance = turn_clearance
                self.backup_updates = 1
                if lateral_direction != 0.0:
                    self.state = "LATERAL_FOR_TURN"
                    self.lateral_escape_direction = lateral_direction
                    return np.array(
                        [0.0, lateral_direction * self.lateral_escape_speed, 0.0],
                        dtype=np.float32,
                    )
                if autonomous and forward_clearance > self.forward_stop_distance:
                    self.state = "ADVANCE_FOR_TURN"
                    return np.array([self.advance_speed, 0.0, 0.0], dtype=np.float32)
                if rear_clearance > self.rear_stop_distance:
                    self.state = "BACKUP_FOR_TURN"
                    return np.array([-self.backup_speed, 0.0, 0.0], dtype=np.float32)
                self.state = "BLOCKED"
                return np.zeros(3, dtype=np.float32)
            self.reset()
            return command

        hazard, distance = self.analyze(lidar)
        # Near a goal, a wall or cabinet can be visible beyond the requested
        # stopping point.  The global map has already validated the goal cell;
        # do not let local avoidance turn away from an obstacle behind it.
        if (
            autonomous
            and goal_distance is not None
            and math.isfinite(distance)
            and distance > float(goal_distance) + self.goal_obstacle_margin
        ):
            hazard = "CLEAR"
            distance = math.inf
        self.hazard = hazard
        self.hazard_distance = distance

        if self.state == "FORWARD":
            if hazard == "CLEAR":
                return self._center_forward_command(
                    command, lidar, lateral_velocity
                )
            # A shallow body-width route is a normal forward correction, not
            # an in-place turn.  Prefer it everywhere in the scene before
            # applying the larger rotating-body envelope or backing up.
            if command[0] > 1.0e-3:
                angle = self._choose_forward_arc(lidar)
                if angle is not None:
                    self.resume_speed = float(command[0])
                    return self._begin_forward_arc(
                        angle, yaw, command, lidar, lateral_velocity
                    )
            turn_clearance = lidar.planar_clearance()
            rear_clearance = self._translation_clearance(lidar, math.pi)
            if (
                turn_clearance < self.turning_radius
                and rear_clearance > self.rear_stop_distance
            ):
                self.state = "BACKUP_FOR_TURN"
                self.hazard = "TURN_SWEEP"
                self.hazard_distance = turn_clearance
                self.backup_updates = 1
                self.resume_speed = float(command[0])
                return np.array([-self.backup_speed, 0.0, 0.0], dtype=np.float32)
            self.state = "AVOID_OBSTACLE" if hazard == "OBSTACLE" else "AVOID_DROP"
            self._set_target_heading(lidar, yaw)
            self.resume_speed = float(command[0])

        heading_error = _angle_difference(self.target_yaw, float(yaw))
        if hazard == "CLEAR" and abs(heading_error) <= self.heading_tolerance:
            self.clear_scans += 1
        else:
            self.clear_scans = 0

        if self.clear_scans >= self.clear_scans_required:
            self.state = "FORWARD"
            self.hazard = "CLEAR"
            self.hazard_distance = math.inf
            return np.array([self.resume_speed, 0.0, 0.0], dtype=np.float32)

        # Reaching a candidate heading that is no longer safe causes a fresh
        # local search instead of forcing a predetermined extra rotation.
        if abs(heading_error) <= self.heading_tolerance and hazard != "CLEAR":
            self._set_target_heading(lidar, yaw)
            heading_error = _angle_difference(self.target_yaw, float(yaw))

        yaw_rate = float(np.clip(1.8 * heading_error, -self.turn_speed, self.turn_speed))
        if abs(yaw_rate) < 0.22 and abs(heading_error) > self.heading_tolerance:
            yaw_rate = math.copysign(0.22, heading_error)
        self.turn_direction = 1.0 if yaw_rate >= 0.0 else -1.0
        return np.array([0.0, 0.0, yaw_rate], dtype=np.float32)

    def status_text(self):
        distance = "--" if not math.isfinite(self.hazard_distance) else f"{self.hazard_distance:.2f}m"
        if self.state == "FORWARD":
            if abs(self.centering_lateral) > 1.0e-3:
                side = "LEFT" if self.centering_lateral > 0.0 else "RIGHT"
                return (
                    f"NAV: CENTER {side}  Vy={self.centering_lateral:+.2f}  "
                    f"path={self.hazard}"
                )
            return f"NAV: FORWARD  path={self.hazard}  distance={distance}"
        if self.state == "BACKUP_FOR_TURN":
            return f"NAV: BACKUP FOR TURN  clearance={distance}"
        if self.state == "ADVANCE_FOR_TURN":
            return f"NAV: ADVANCE FOR TURN  clearance={distance}"
        if self.state == "LATERAL_FOR_TURN":
            side = "LEFT" if self.lateral_escape_direction > 0.0 else "RIGHT"
            return f"NAV: MOVE {side} FOR TURN  clearance={distance}"
        if self.state == "BLOCKED":
            return f"NAV: BLOCKED  clearance={distance}"
        if self.state == "STEER_FORWARD":
            side = "LEFT" if self.turn_direction > 0.0 else "RIGHT"
            target_deg = math.degrees(self.target_relative_angle)
            return (
                f"NAV: STEER FORWARD  turn={side}  "
                f"target={target_deg:+.0f}deg  distance={distance}"
            )
        side = "LEFT" if self.turn_direction > 0.0 else "RIGHT"
        target_deg = math.degrees(self.target_relative_angle)
        return (
            f"NAV: {self.state}  turn={side}  "
            f"target={target_deg:+.0f}deg  distance={distance}"
        )
