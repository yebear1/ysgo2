"""Contract and fault tests; fake clock and SDK-shaped client, no DDS."""
import contextlib
from dataclasses import replace
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.adapters import EduSportSink, MujocoSensorAdapter
from navigation.contracts import VelocityRequest
from navigation.motion_gateway import MotionGateway
from navigation.run_offline import setup, scenario, FAULTS
from navigation.runtime import PlanarSafety


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.client, self.world, self.runtime = setup()
        self.frame = self.world.frame()
        self.gate = self.runtime.gateway
        self.epoch = self.gate.arm(self.frame)

    def send(self, frame=None, velocity=(.1, 0., 0.), stamp=None, epoch=None):
        return self.gate.dispatch(frame or self.frame, autonomous=VelocityRequest(
            self.clock() if stamp is None else stamp, velocity,
            self.epoch if epoch is None else epoch))

    def test_limits_before_swept_check(self):
        seen = []
        self.gate.safety_filter = lambda command, scan: seen.append(command) or True
        d = self.send(velocity=(2., -2., 3.))
        self.assertEqual(seen, [(.1, -.1, .2)])
        self.assertEqual(d.velocity, seen[0])
        self.assertEqual(self.client.velocity, d.velocity)

    def test_zero_manual_takes_over_and_never_auto_resumes(self):
        self.assertEqual(self.send().source, 'autonomy')
        d = self.gate.dispatch(self.frame, manual=VelocityRequest(0., (0.,0.,0.)))
        self.assertEqual(d.source, 'stop')
        self.assertEqual(self.send().reason, 'autonomy_not_armed')
        new_epoch = self.gate.arm(self.frame)
        self.assertNotEqual(new_epoch, self.epoch)
        self.assertEqual(self.send().reason, 'autonomy_lease_invalid')

    def test_manual_can_work_without_pose_but_requires_scan(self):
        frame = replace(self.frame, pose=None)
        d = self.gate.dispatch(frame, manual=VelocityRequest(0., (0., .04, 0.)))
        self.assertEqual(d.source, 'manual')
        d = self.gate.dispatch(replace(frame, scan=None), manual=VelocityRequest(0., (.04,0.,0.)))
        self.assertEqual(d.reason, 'scan_unavailable')

    def test_future_and_expired_request(self):
        for stamp in (1., -.16, math.nan):
            self.gate.arm(self.frame); self.epoch = self.gate.epoch
            self.assertEqual(self.send(stamp=stamp).reason, 'command_expired')
        self.assertFalse(self.client.moves)

    def test_expired_unchanged_heartbeat(self):
        self.send()
        self.clock.advance(.21)
        frame = replace(self.world.frame(), robot=self.frame.robot)
        d = self.send(frame)
        self.assertEqual(d.reason, 'robot_state_unavailable')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_replayed_sequence_cannot_refresh_time(self):
        self.clock.advance(.05)
        frame = replace(self.frame, robot=replace(self.frame.robot, stamp=.05))
        self.assertEqual(self.send(frame).reason, 'robot_sequence_invalid')

    def test_invalid_velocity_stops_even_after_previous_move(self):
        self.send()
        self.assertEqual(self.send(velocity=('bad',0.,0.)).source, 'stop')
        self.assertEqual(self.client.velocity, (0.,0.,0.))
        self.assertIsNotNone(self.gate.fault)

    def test_clock_regression_latches(self):
        self.send()
        self.clock.time = -.01
        self.assertEqual(self.send().reason, 'clock_regressed')

    def test_fault_reset_requires_disabled_and_explicit_rearm(self):
        emergency = replace(self.frame, robot=replace(self.frame.robot, emergency=True))
        self.assertEqual(self.send(emergency).reason, 'emergency_latched')
        with self.assertRaises(ValueError): self.gate.reset_fault(self.frame)
        disabled = replace(self.frame, robot=replace(self.frame.robot, enabled=False))
        self.gate.reset_fault(disabled)
        self.assertIsNone(self.gate.fault)
        self.assertFalse(self.gate.armed)
        self.assertEqual(self.send().reason, 'autonomy_not_armed')
        self.epoch = self.gate.arm(self.frame)
        self.assertEqual(self.send().source, 'autonomy')

    def test_stop_failure_must_not_be_reported_as_stopped(self):
        self.send()
        self.client.stop_code = -1
        d = self.gate.stop()
        self.assertEqual(d.reason, 'stop_not_acknowledged')
        self.assertFalse(d.transport_ok)
        self.assertFalse(d.physical_stop_verified)
        disabled = replace(self.frame, robot=replace(self.frame.robot, enabled=False))
        with self.assertRaises(ValueError): self.gate.reset_fault(disabled)
        self.assertIsNotNone(self.gate.fault)
        self.client.stop_code = 0
        self.gate.reset_fault(disabled)
        self.assertIsNone(self.gate.fault)

    def test_processing_deadline_checks_pose_and_scan_again(self):
        for source in ('pose','scan','command'):
            clock, client, world, runtime = setup()
            clock.time = 1.
            frame = world.frame()
            if source == 'pose': frame = replace(frame, pose=replace(frame.pose, stamp=.76))
            if source == 'scan': frame = replace(frame, scan=replace(frame.scan, stamp=.81))
            epoch = runtime.gateway.arm(frame)
            def slow(*_): clock.advance(.16 if source == 'command' else .03); return True
            runtime.gateway.safety_filter = slow
            d = runtime.gateway.dispatch(frame, autonomous=VelocityRequest(1., (.1,0.,0.), epoch))
            self.assertEqual(d.reason, 'processing_deadline_missed')
            self.assertFalse(client.moves)

    def test_pose_expiring_during_send_triggers_stop_on_return(self):
        self.clock.time = 1.
        frame = self.world.frame()
        frame = replace(frame, pose=replace(frame.pose, stamp=.76))
        self.epoch = self.gate.arm(frame)
        self.client.delay = .03
        self.assertEqual(self.send(frame).reason, 'transport_deadline_missed')
        self.assertEqual(len(self.client.moves), 1)
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_invalid_scan_or_geometry_is_not_free_space(self):
        check = PlanarSafety()
        for scan in (replace(self.frame.scan, frame_id='camera'),
                     replace(self.frame.scan, ranges=(math.inf,)*120),
                     replace(self.frame.scan, ranges=(-1.,)*120),
                     replace(self.frame.scan, angles=(0.,)*120),
                     replace(self.frame.scan, max_range=.5)):
            self.assertFalse(check((.1,0.,0.), scan))

    def test_first_stage_pulse_has_explicit_unscanned_manual_profile(self):
        gate = MotionGateway(EduSportSink(self.client), self.clock, allow_unscanned_manual=True)
        frame = replace(self.frame, pose=None, scan=None)
        d = gate.dispatch(frame, manual=VelocityRequest(0., (.05,0.,0.)))
        self.assertEqual(d.source, 'manual')
        with self.assertRaises(ValueError): gate.arm(frame)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.client, self.world, self.runtime = setup()
        self.frame = self.world.frame()

    def start(self, goal=(3.,0.), yaw=None):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.runtime.set_goal(goal, self.frame, yaw))
        self.runtime.arm(self.frame)
        self.assertEqual(self.runtime.step(self.frame).source, 'autonomy')

    def test_unmatched_pose_never_arms(self):
        frame = replace(self.frame, pose=replace(self.frame.pose, matched=False))
        with self.assertRaises(ValueError): self.runtime.set_goal((3.,0.), frame)
        self.assertFalse(self.client.moves)

    def test_manual_cancels_route_and_release_waits(self):
        self.start()
        d = self.runtime.step(self.frame, VelocityRequest(0., (0.,.05,0.)))
        self.assertEqual(d.source, 'manual')
        self.assertFalse(self.runtime.planner.active)
        self.assertEqual(self.runtime.step(self.frame).reason, 'autonomy_not_armed')

    def test_map_session_replacement_cancels_motion(self):
        self.start()
        self.runtime.update_map('another-map', self.world.grid())
        self.assertFalse(self.runtime.gateway.armed)
        self.assertFalse(self.runtime.planner.active)
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_bad_map_cancels_running_goal(self):
        self.start()
        grid = self.world.grid(); grid['data'] = np.full((100,120), math.nan)
        with self.assertRaises(ValueError): self.runtime.update_map('mock-map', grid)
        self.assertFalse(self.runtime.planner.map_ready)
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_dynamic_wall_stops_path(self):
        self.start()
        grid = self.world.grid(); grid['data'][:, 50:52] = 100
        self.runtime.update_map('mock-map', grid)
        self.assertEqual(self.runtime.step(self.frame).source, 'stop')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_exact_requested_point_not_nearest_cell(self):
        goal = (.024, .024)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(self.runtime.set_goal(goal, self.frame))
        np.testing.assert_allclose(self.runtime.planner.effective_goal, goal)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.runtime.set_goal((1.5, 1.2), self.frame))

    def test_terminal_checks_final_yaw_before_success(self):
        self.start((.05,0.), yaw=math.pi/2)
        self.assertFalse(self.runtime.planner.reached)
        self.assertGreater(self.client.velocity[2], 0.)
        self.world.pose[:] = [.05,0.,math.pi/2]
        frame = self.world.frame()
        self.assertEqual(self.runtime.step(frame).reason, 'goal_reached')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_terminal_does_not_bypass_updated_unknown_map(self):
        self.start((.6,0.), yaw=0.)
        grid = self.world.grid()
        grid['data'][:, 28:30] = -1
        self.runtime.update_map('mock-map', grid)
        self.assertEqual(self.runtime.step(self.frame).reason, 'terminal_path_not_observed_free')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_failed_rearm_stops_previous_command(self):
        self.start()
        with self.assertRaises(ValueError):
            self.runtime.gateway.arm(replace(self.frame, scan=None))
        self.assertEqual(self.client.velocity, (0.,0.,0.))
        self.assertFalse(self.runtime.gateway.armed)

    def test_emergency_latches_even_with_missing_pose(self):
        self.start()
        frame = replace(self.frame, pose=None, robot=replace(self.frame.robot, emergency=True))
        self.assertEqual(self.runtime.step(frame).reason, 'emergency_latched')
        self.assertEqual(self.runtime.gateway.fault, 'emergency_latched')

    def test_pose_outside_map_cannot_be_clamped_into_a_route(self):
        self.start()
        frame = replace(self.frame, pose=replace(self.frame.pose, xy_yaw=(50.,0.,0.)))
        self.assertEqual(self.runtime.step(frame).reason, 'pose_outside_observed_map')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_planner_exception_stops_previous_motion(self):
        self.start()
        def broken(*_): raise RuntimeError('planner error')
        self.runtime.planner.update = broken
        self.assertEqual(self.runtime.step(self.frame).reason, 'planner_error')
        self.assertEqual(self.client.velocity, (0.,0.,0.))

    def test_mujoco_adapter_never_substitutes_truth_for_pose(self):
        adapter = MujocoSensorAdapter('saved-map')
        bridge = NS(global_pose=None, navigation_velocity=(0.,0.,0.))
        frame = adapter.read(0., bridge, None, True)
        self.assertIsNone(frame.pose)
        self.assertIsNone(frame.scan)
        self.assertEqual(frame.clock_id, 'simulation')
        bridge.global_pose = [1.,2.,3.]
        bridge.global_pose_timestamp, bridge.map_match_verified = -.5, False
        frame = adapter.read(0., bridge, None, True)
        self.assertEqual(frame.pose.stamp, -.5)
        self.assertFalse(frame.pose.matched)

    def test_shared_sim_session_uses_gateway_and_does_not_resume(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'deploy_mujoco'))
        from shared_navigation import SharedNavigationSession
        session = SharedNavigationSession(self.clock)
        session.update_map(self.world.grid())
        bridge = NS(global_pose=[0.,0.,0.], global_pose_timestamp=0.,
                    map_match_verified=True, navigation_velocity=(0.,0.,0.))
        scan = self.frame.scan
        lidar = NS(scan_stamp=0., planar_angles=scan.angles, planar_ranges=scan.ranges, planar_max_range=4.)
        session.request_goal((3.,0.))
        self.assertEqual(session.step(bridge, lidar).source, 'autonomy')
        bridge.global_pose = None
        self.assertEqual(session.step(bridge, lidar).source, 'stop')
        bridge.global_pose = [0.,0.,0.]
        self.assertEqual(session.step(bridge, lidar).reason, 'autonomy_not_armed')
        session.request_goal((3.,0.))
        self.assertEqual(session.step(bridge, lidar).source, 'autonomy')
        session.step(bridge, lidar, manual=(0.,0.,0.))
        self.assertEqual(session.step(bridge, lidar).reason, 'autonomy_not_armed')

    def test_core_imports_without_simulator_ros_torch_or_sdk(self):
        code = """import sys
sys.modules.update({x: None for x in ('mujoco','torch','rclpy','unitree_sdk2py')})
from navigation.runtime import NavigationRuntime
from navigation.adapters import EduSportSink
"""
        subprocess.run([sys.executable, '-c', code], check=True,
                       cwd=str(Path(__file__).resolve().parents[1]))

    def test_fault_scenarios(self):
        for name in FAULTS:
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(scenario(name)['passed'])


if __name__ == '__main__':
    unittest.main()
