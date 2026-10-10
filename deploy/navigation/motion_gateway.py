"""Single-writer arbitration with explicit rearming and bounded command leases.

No SDK/ROS/NumPy imports. The injected synchronous sink must have bounded calls;
this is not an independent hardware watchdog and cannot interrupt a hung SDK.
"""
import math
import threading
from .contracts import Decision

ZERO = (0., 0., 0.)


class MotionGateway:
    def __init__(self, sink, clock, safety_filter=None, limits=(.10, .10, .20),
                 clock_id='monotonic', allow_unscanned_manual=False):
        if len(limits) != 3 or not all(math.isfinite(v) and v > 0 for v in limits):
            raise ValueError('Invalid command limits')
        self.sink, self.clock, self.safety_filter = sink, clock, safety_filter
        self.limits, self.clock_id = tuple(limits), clock_id
        self.allow_unscanned_manual = allow_unscanned_manual
        self.armed, self.epoch, self.fault = False, 0, None
        self._last_now, self._last_robot = None, None
        self._stopped = False
        self._lock = threading.RLock()
        self.last = Decision('stop', ZERO, 'not_started', False)

    @staticmethod
    def fresh(stamp, now, timeout):
        try:
            return math.isfinite(now) and math.isfinite(stamp) and 0 <= now-stamp <= timeout
        except (TypeError, ValueError):
            return False

    def _health(self, frame, now):
        if not math.isfinite(now) or (self._last_now is not None and now < self._last_now):
            self.fault = 'clock_regressed'
        self._last_now = now
        if frame.clock_id != self.clock_id:
            self.fault = 'clock_domain_mismatch'
        r = frame.robot
        if self._last_robot is not None:
            seq, stamp = self._last_robot
            if r.sequence < seq or r.stamp < stamp or (r.sequence == seq and r.stamp != stamp):
                self.fault = 'robot_sequence_invalid'
        if (not isinstance(r.sequence, int) or isinstance(r.sequence, bool) or r.sequence < 0
                or not math.isfinite(r.stamp)):
            self.fault = 'robot_sequence_invalid'
        self._last_robot = (r.sequence, r.stamp)
        if r.emergency:
            self.fault = 'emergency_latched'
        if self.fault:
            return self.fault
        if not r.connected or not self.fresh(r.stamp, now, .20):
            return 'robot_state_unavailable'
        if not r.enabled:
            return 'operator_disabled'
        return None

    def check_health(self, frame):
        with self._lock:
            try:
                reason = self._health(frame, self.clock())
            except Exception:
                self.fault = reason = 'invalid_input'
            return self._stop(reason) if reason else None

    def _pose_ready(self, frame, now):
        p = frame.pose
        return (p is not None and p.matched and bool(p.map_id) and p.frame_id == 'map'
                and len(p.xy_yaw) == 3 and all(math.isfinite(v) for v in p.xy_yaw)
                and self.fresh(p.stamp, now, .25))

    def disarm(self):
        with self._lock:
            self.armed = False
            self.epoch += 1

    def arm(self, frame):
        with self._lock:
            try:
                now = self.clock()
                reason = self._health(frame, now)
                if reason or not self._pose_ready(frame, now):
                    raise ValueError(reason or 'verified_map_pose_required')
                if frame.scan is None or not self.fresh(frame.scan.stamp, now, .20):
                    raise ValueError('fresh_scan_required')
                self.epoch += 1
                self.armed = True
                return self.epoch
            except Exception:
                self._stop('arming_rejected')
                raise

    def reset_fault(self, frame):
        """Explicit operator reset while disabled, with fresh robot status."""
        with self._lock:
            now = self.clock()
            if (frame.clock_id != self.clock_id or frame.robot.enabled or frame.robot.emergency
                    or not frame.robot.connected or not self.fresh(frame.robot.stamp, now, .20)):
                raise ValueError('Reset requires fresh, disabled, non-emergency robot state')
            if not self._stop('operator_reset').transport_ok:
                raise ValueError('Stop not acknowledged; fault retained')
            self.fault = None
            self._last_now, self._last_robot = now, (frame.robot.sequence, frame.robot.stamp)
            self.disarm()

    def _stop(self, reason):
        self.disarm()
        ok = self._stopped
        if not ok:
            for _ in range(3):
                try:
                    if self.sink.stop():
                        ok = True
                except Exception:
                    pass
            self._stopped = ok
        if not ok:
            self.fault = 'stop_not_acknowledged'
        self.last = Decision('stop', ZERO, self.fault or reason, ok)
        return self.last

    def stop(self, reason='shutdown'):
        with self._lock:
            return self._stop(reason)

    def dispatch(self, frame, autonomous=None, manual=None):
        with self._lock:
            try:
                return self._dispatch(frame, autonomous, manual)
            except Exception:
                self.fault = 'invalid_input'
                return self._stop(self.fault)

    def _dispatch(self, frame, autonomous=None, manual=None):
        with self._lock:
            now = self.clock()
            reason = self._health(frame, now)
            if reason:
                return self._stop(reason)
            if manual is not None:
                self.disarm()  # Even zero/manual takeover invalidates queued autonomy.
                source, request = 'manual', manual
            else:
                source, request = 'autonomy', autonomous
                if not self.armed:
                    return self._stop('autonomy_not_armed')
                if not self._pose_ready(frame, now):
                    return self._stop('localization_unavailable')
                if request is None or request.epoch != self.epoch:
                    return self._stop('autonomy_lease_invalid')
            if request is None or not self.fresh(request.stamp, now, .15):
                return self._stop('command_expired')
            if len(request.velocity) != 3 or not all(math.isfinite(v) for v in request.velocity):
                self.fault = 'invalid_velocity'
                return self._stop(self.fault)
            command = tuple(max(-cap, min(cap, v)) for v, cap in zip(request.velocity, self.limits))
            if command == ZERO:
                return self._stop('zero_request')
            unscanned = source == 'manual' and self.allow_unscanned_manual
            if not unscanned:
                if frame.scan is None or not self.fresh(frame.scan.stamp, now, .20):
                    return self._stop('scan_unavailable')
                # Limits are applied BEFORE swept-path checking; no later reshaping.
                try:
                    clear = self.safety_filter is not None and self.safety_filter(command, frame.scan)
                except Exception:
                    clear = False
                if not clear:
                    return self._stop('motion_not_clear')
            checked_at = self.clock()
            if (not self.fresh(request.stamp, checked_at, .15)
                    or not self.fresh(frame.robot.stamp, checked_at, .20)
                    or (source == 'autonomy' and not self._pose_ready(frame, checked_at))
                    or (not unscanned and not self.fresh(frame.scan.stamp, checked_at, .20))):
                return self._stop('processing_deadline_missed')
            self._stopped = False  # A send can act on the robot even if it then throws.
            try:
                ok = self.sink.send(command)
            except Exception:
                ok = False
            if not ok:
                self.fault = 'transport_failure'
                return self._stop(self.fault)
            returned_at = self.clock()
            if (not self.fresh(request.stamp, returned_at, .15)
                    or not self.fresh(frame.robot.stamp, returned_at, .20)
                    or (source == 'autonomy' and not self._pose_ready(frame, returned_at))
                    or (not unscanned and not self.fresh(frame.scan.stamp, returned_at, .20))):
                self.fault = 'transport_deadline_missed'
                return self._stop(self.fault)
            self.last = Decision(source, command, 'sent', True)
            return self.last
