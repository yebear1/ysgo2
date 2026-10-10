"""Local tests only: fake clocks and clients; never initializes DDS."""
import contextlib
import io
import math
import struct
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from go2_edu_bringup import (A, R1, SELECT, GuardError, StateMonitor, main,
                             run_pulse, validate_pulse)


class Clock:
    time = 0.
    def __call__(self):
        return self.time
    def sleep(self, duration):
        self.time += duration


class Client:
    def __init__(self, clock):
        self.clock, self.moves, self.stops = clock, [], 0
        self.move_code, self.stop_code = 0, 0
    def Move(self, *cmd):
        self.moves.append((self.clock(), cmd))
        return self.move_code
    def StopMove(self):
        self.stops += 1
        return self.stop_code


def state(buttons=0, tilt=0):
    return {'buttons': buttons, 'tilt_degrees': tilt, 'axes': [0., 0., 0., 0.]}


def packet(tick=1):
    return NS(tick=tick, imu_state=NS(quaternion=[1., 0., 0., 0.],
                                    gyroscope=[0.]*3, accelerometer=[0.,0.,9.81]),
              motor_state=[NS(q=0., dq=0., lost=0) for _ in range(20)],
              wireless_remote=[0]*40, foot_force=[0]*4)


class BringupTests(unittest.TestCase):
    def setup_pulse(self, snapshot=None):
        c = Clock()
        client = Client(c)
        monitor = NS(snapshot=snapshot or (lambda: state(0 if c()<.1 else R1|A)))
        mode = NS(CheckMode=lambda: (0, {'name': 'normal'}))
        return c, client, monitor, mode

    def pulse(self, c, client, monitor, mode, **kwargs):
        original_snapshot = monitor.snapshot
        sequence = [0]
        def timed_snapshot():
            packet = original_snapshot()
            sequence[0] += 1
            packet.update(time=c(), sequence=sequence[0])
            return packet
        monitor.snapshot = timed_snapshot
        with contextlib.redirect_stdout(io.StringIO()):
            return run_pulse(monitor, client, mode, 'vx', .05, 1., 'normal',
                             clock=c, sleep=c.sleep, **kwargs)

    def test_bounded_pulse_and_stop(self):
        c, cl, m, mode = self.setup_pulse()
        result = self.pulse(c, cl, m, mode)
        self.assertTrue(result['pulse_completed'])
        self.assertFalse(result['physical_stop_verified'])
        self.assertTrue(cl.moves)
        self.assertLessEqual(cl.moves[-1][0]-cl.moves[0][0], 1.)
        self.assertTrue(all(cmd == (.05,0.,0.) for _,cmd in cl.moves))
        self.assertEqual(cl.stops, 3)

    def test_held_buttons_do_not_auto_arm(self):
        c, cl, m, mode = self.setup_pulse(lambda: state(R1|A))
        with self.assertRaisesRegex(GuardError, 'Arming timed out'):
            self.pulse(c, cl, m, mode)
        self.assertFalse(cl.moves)
        self.assertEqual(cl.stops, 0)

    def test_mode_mismatch_does_not_move(self):
        c, cl, m, mode = self.setup_pulse()
        mode.CheckMode = lambda: (0, {'name': 'other'})
        with self.assertRaisesRegex(GuardError, 'mode mismatch'):
            self.pulse(c, cl, m, mode)
        self.assertFalse(cl.moves)

    def test_mode_change_while_arming_prevents_motion(self):
        c, cl, m, mode = self.setup_pulse()
        replies = iter([(0, {'name': 'normal'}), (0, {'name': 'other'})])
        mode.CheckMode = lambda: next(replies)
        with self.assertRaisesRegex(GuardError, 'changed while arming'):
            self.pulse(c, cl, m, mode)
        self.assertFalse(cl.moves)

    def test_stock_python_gets_clear_version_error(self):
        with patch('sys.version_info', (3, 6, 9)), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['check']), 2)

    def test_release_select_tilt_and_stale_stop(self):
        for failure in ['release','select','tilt','stale']:
            with self.subTest(failure=failure):
                c, cl, m, mode = self.setup_pulse()
                def snapshot():
                    if c()<.1: return state()
                    if c()<.3: return state(R1|A)
                    if failure=='stale': raise GuardError('stale')
                    if failure=='release': return state()
                    if failure=='select': return state(R1|A|SELECT)
                    return state(R1|A, 30)
                m.snapshot = snapshot
                with self.assertRaises(GuardError): self.pulse(c, cl, m, mode)
                self.assertTrue(cl.moves)
                self.assertEqual(cl.stops, 3)
                self.assertLess(c(), .5)

    def test_send_failure_and_exception_stop(self):
        for exception in [False, True]:
            c, cl, m, mode = self.setup_pulse()
            if exception:
                def broken(*args): raise RuntimeError('transport broke')
                cl.Move = broken
            else: cl.move_code = -1
            with self.assertRaises((GuardError,RuntimeError)):
                self.pulse(c, cl, m, mode)
            self.assertEqual(cl.stops, 3)

    def test_missing_stop_ack_is_not_success(self):
        c, cl, m, mode = self.setup_pulse()
        cl.stop_code = -1
        with self.assertRaisesRegex(GuardError, 'NOT acknowledged'):
            self.pulse(c, cl, m, mode)

    def test_limits_reject_nan_and_out_of_range(self):
        for axis,speed,duration in [('vx',.11,1),('yaw',.21,1),('vy',math.nan,1),
                                    ('vx',0,1),('vx',.05,3),('vx',.05,math.inf)]:
            with self.assertRaises(GuardError): validate_pulse(axis,speed,duration)

    def test_duplicate_packets_do_not_refresh(self):
        c = Clock(); m = StateMonitor(c); m.receive(packet())
        c.time = .21; m.receive(packet())
        with self.assertRaisesRegex(GuardError, 'stale'): m.snapshot()
        m.receive(packet(2)); self.assertEqual(m.snapshot()['tick'], 2)

    def test_tick_wrap_and_regression(self):
        c=Clock(); m=StateMonitor(c); m.receive(packet(0xffffffff)); m.receive(packet(0))
        self.assertEqual(m.snapshot()['tick'], 0)
        m.receive(packet(0xffffffff))
        with self.assertRaisesRegex(GuardError, 'backwards'): m.snapshot()

    def test_invalid_packet_latches(self):
        for field in ['quat','motor','remote']:
            m=StateMonitor(); p=packet()
            if field=='quat': p.imu_state.quaternion=[0]*4
            elif field=='motor': p.motor_state[0].dq=math.nan
            else: p.wireless_remote=[0]*3
            m.receive(p);m.receive(packet(2))
            with self.assertRaises(GuardError):m.snapshot()

    def test_motion_cli_requires_enable_before_dds(self):
        with patch('socket.if_nameindex',return_value=[(1,'eth0')]), \
             patch.dict('sys.modules',{'unitree_sdk2py.core.channel':None}), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['pulse','--net','eth0','--axis','vx','--speed','.05','--expected-mode','normal'])
            self.assertEqual(error.exception.code,2)

    def test_command_helpers_clear_declared_dq_field(self):
        # Match the SDK MotorCmd schema; slots reject the old undeclared qd.
        from common.command_helper import create_zero_cmd, create_damping_cmd
        class Motor:
            __slots__ = ('q', 'dq', 'kp', 'kd', 'tau')
        for helper in [create_zero_cmd, create_damping_cmd]:
            cmd = NS(motor_cmd=[Motor() for _ in range(20)])
            for motor in cmd.motor_cmd: motor.dq = 123.
            helper(cmd)
            self.assertTrue(all(m.dq == 0 for m in cmd.motor_cmd))


if __name__ == '__main__':
    unittest.main()
