#!/usr/bin/env python3
"""Go2 EDU first-stage tools. No DDS initialization until listen/pulse is selected.

Python 3.8+, no Torch, ROS, or CUDA dependency. pulse uses the OEM Sport API,
never low-level motor commands or automatic standing/mode switching.
"""
import argparse
import importlib
import json
import math
import platform
import socket
import struct
import sys
import threading
import time
from pathlib import Path

# Shared gateway is stdlib-only; keep Python 3.6 version diagnostics available.
if sys.version_info >= (3, 8):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from navigation.adapters import EduSportSink
    from navigation.contracts import NavigationFrame, RobotState, VelocityRequest
    from navigation.motion_gateway import MotionGateway

R1, A, SELECT = 1 << 0, 1 << 8, 1 << 3
FRESHNESS = 0.20


class GuardError(RuntimeError):
    pass


def finite(values):
    return all(math.isfinite(float(v)) for v in values)


class StateMonitor:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.state = None
        self.error = None
        self.count = 0

    def receive(self, msg):
        try:
            q = list(msg.imu_state.quaternion)
            gyro = list(msg.imu_state.gyroscope)
            accel = list(msg.imu_state.accelerometer)
            motors = list(msg.motor_state[:12])
            remote = bytes(msg.wireless_remote)
            if len(q) != 4 or len(gyro) != 3 or len(accel) != 3 or len(motors) != 12 or len(remote) != 40:
                raise GuardError('Invalid LowState dimensions')
            positions = [m.q for m in motors]
            velocities = [m.dq for m in motors]
            axes = struct.unpack_from('<ffff', remote, 4)[:3] + (struct.unpack_from('<f', remote, 20)[0],)
            if not finite(q + gyro + accel + positions + velocities + list(axes)):
                raise GuardError('Non-finite sensor/remote data')
            norm = math.sqrt(sum(v*v for v in q))
            if not 0.9 <= norm <= 1.1 or any(abs(v) > 1.1 for v in axes):
                raise GuardError('Invalid IMU quaternion or remote axes')
            if any(m.lost for m in motors):
                raise GuardError('Motor state reports communication loss')
            w, x, y, z = [v/norm for v in q]
            tilt = math.degrees(math.acos(max(-1., min(1., 1 - 2*(x*x+y*y)))))
            state = dict(tick=int(msg.tick), time=self.clock(),
                         buttons=struct.unpack_from('<H', remote, 2)[0],
                         axes=list(axes), tilt_degrees=tilt,
                         quaternion=q, gyroscope=gyro, accelerometer=accel,
                         joint_positions=positions, joint_velocities=velocities,
                         foot_force=list(msg.foot_force))
            with self.lock:
                if self.state is not None:
                    delta = (state['tick'] - self.state['tick']) & 0xffffffff
                    if delta == 0:
                        return  # Repeated packets must not refresh freshness.
                    if delta >= 0x80000000:
                        raise GuardError('LowState tick moved backwards; restart inspection')
                self.state = state
                self.count += 1
                self.state['sequence'] = self.count
        except Exception as exc:
            with self.lock:
                self.error = str(exc)  # Latched: a motion session never auto-resumes.

    def snapshot(self):
        with self.lock:
            if self.error:
                raise GuardError(self.error)
            if self.state is None:
                raise GuardError('No LowState received')
            state = dict(self.state)
        age = self.clock() - state['time']
        if age < 0 or age > FRESHNESS:
            raise GuardError('LowState stale: %.3f seconds' % age)
        return state


def motion_guard(state):
    if state['buttons'] & SELECT:
        raise GuardError('Select pressed')
    if state['tilt_degrees'] > 20:
        raise GuardError('Body tilt exceeds the 20 degree bring-up limit')
    if any(abs(v) > .08 for v in state['axes']):
        raise GuardError('Remote sticks must be centered')


def validate_pulse(axis, speed, seconds):
    if axis not in ('vx', 'vy', 'yaw'):
        raise GuardError('Unknown motion axis')
    cap = .20 if axis == 'yaw' else .10
    if not math.isfinite(speed) or not 0 < abs(speed) <= cap:
        raise GuardError('Speed must be nonzero and at most %.2f in magnitude' % cap)
    if not math.isfinite(seconds) or not 0 < seconds <= 2:
        raise GuardError('Pulse duration must be in (0, 2] seconds')


def run_pulse(monitor, sport, switcher, axis, speed, seconds, expected_mode,
              clock=time.monotonic, sleep=time.sleep):
    validate_pulse(axis, speed, seconds)
    if not expected_mode:
        raise GuardError('An operator-verified OEM mode name is required')
    code, mode = switcher.CheckMode()
    if code != 0 or not isinstance(mode, dict) or mode.get('name') != expected_mode:
        raise GuardError('OEM mode mismatch: %r; no mode was changed' % (mode,))
    print('Release R1 and A, then hold R1+A within 15s. Select or release stops the pulse.', flush=True)
    deadline = clock() + 15
    released = False
    while True:
        state = monitor.snapshot()
        motion_guard(state)
        buttons = state['buttons']
        if not buttons & (R1 | A):
            released = True
        if released and buttons & (R1 | A) == R1 | A:
            break
        if clock() >= deadline:
            raise GuardError('Arming timed out; no motion command sent')
        sleep(.05)
    code, mode = switcher.CheckMode()
    if code != 0 or not isinstance(mode, dict) or mode.get('name') != expected_mode:
        raise GuardError('OEM mode changed while arming; no motion command sent')
    command = [0., 0., 0.]
    command[('vx', 'vy', 'yaw').index(axis)] = speed
    # Only this manually supervised pulse may run without range sensors.
    gateway = MotionGateway(EduSportSink(sport), clock, allow_unscanned_manual=True)
    attempted = False
    deadline = clock() + seconds
    try:
        while clock() < deadline:
            state = monitor.snapshot()
            motion_guard(state)
            if state['buttons'] & (R1 | A) != R1 | A:
                raise GuardError('R1/A released')
            attempted = True
            frame = NavigationFrame(RobotState(state['time'], state['sequence'], True, True))
            decision = gateway.dispatch(frame, manual=VelocityRequest(clock(), tuple(command)))
            if decision.reason != 'sent':
                raise GuardError('Motion gateway stopped: ' + decision.reason)
            sleep(min(.05, max(0., deadline-clock())))
    finally:
        if attempted:
            if not gateway.stop('pulse_finished').transport_ok:
                raise GuardError('StopMove NOT acknowledged; use the independent operator stop')
    return {'pulse_completed': True, 'axis': axis, 'speed': speed, 'requested_seconds': seconds,
            'physical_stop_verified': False}


def offline_check():
    result = {'python': platform.python_version(), 'architecture': platform.machine(),
              'interfaces': [name for _, name in socket.if_nameindex()],
              'dds_initialized': False, 'hardware_verified': False, 'dependencies': {}}
    good = sys.version_info >= (3, 8)
    for name in ('unitree_sdk2py.core.channel', 'unitree_sdk2py.go2.sport.sport_client',
                 'unitree_sdk2py.comm.motion_switcher.motion_switcher_client'):
        try:
            importlib.import_module(name)
            result['dependencies'][name] = 'import OK'
        except Exception as exc:
            good = False
            result['dependencies'][name] = str(exc)
    result['local_import_check_passed'] = good
    return result, 0 if good else 2


def main(argv=None):
    if sys.version_info < (3, 8):
        print('Python 3.8+ is required. collect_tx1_info.py can run on stock Python 3.6.', file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('check', help='Offline imports only; no network or motion')
    for op in ('listen', 'pulse'):
        p = sub.add_parser(op)
        p.add_argument('--net', required=True, help='Explicit robot-facing interface; no autodetection')
        p.add_argument('--seconds', type=float, default=10. if op == 'listen' else 1.)
        if op == 'listen':
            p.add_argument('--query-mode', action='store_true', help='Also query OEM mode via read-only RPC')
        else:
            p.add_argument('--enable-motion', action='store_true')
            p.add_argument('--expected-mode', required=True)
            p.add_argument('--axis', choices=('vx', 'vy', 'yaw'), required=True)
            p.add_argument('--speed', type=float, required=True)
    args = parser.parse_args(argv)
    if args.operation == 'check':
        result, code = offline_check()
        print(json.dumps(result, indent=2))
        return code
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 60:
        parser.error('--seconds must be finite and in (0, 60]')
    if args.net == 'lo' or args.net not in dict((n, i) for i, n in socket.if_nameindex()):
        parser.error('Choose an existing non-loopback robot-facing interface')
    if args.operation == 'pulse':
        if not args.enable_motion:
            parser.error('pulse requires --enable-motion; no DDS or motion was started')
        validate_pulse(args.axis, args.speed, args.seconds)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, args.net)
    monitor = StateMonitor()
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init(monitor.receive, 1)
    try:
        deadline = time.monotonic() + 5
        while monitor.count == 0 and monitor.error is None and time.monotonic() < deadline:
            time.sleep(.02)
        monitor.snapshot()
        switcher = None
        if args.operation == 'pulse' or args.query_mode:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            switcher = MotionSwitcherClient()
            switcher.SetTimeout(.20)
            switcher.Init()
        if args.operation == 'listen':
            if switcher is not None:
                print('OEM mode query: %r' % (switcher.CheckMode(),))
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                print(json.dumps(monitor.snapshot()), flush=True)
                time.sleep(.1)
        else:
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            sport = SportClient()
            sport.SetTimeout(.20)
            sport.Init()
            print(json.dumps(run_pulse(monitor, sport, switcher, args.axis, args.speed,
                                      args.seconds, args.expected_mode)))
    finally:
        subscriber.Close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (GuardError, ImportError, KeyboardInterrupt) as exc:
        print('STOP: %s' % exc, file=sys.stderr)
        sys.exit(2)
