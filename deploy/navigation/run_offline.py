#!/usr/bin/env python3
"""Run reproducible mock EDU scenarios. This entry point never imports a robot SDK."""
import argparse
from dataclasses import replace, asdict
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navigation.adapters import EduSportSink
from navigation.contracts import VelocityRequest
from navigation.mock_edu import FakeClock, MockEduClient, FlatWorld
from navigation.runtime import NavigationRuntime

FAULTS = ('robot_dropout', 'pose_stale', 'unmatched_map', 'scan_dropout', 'invalid_scan',
          'partial_scan', 'obstacle', 'operator_release', 'emergency', 'manual_takeover',
          'send_failure', 'send_exception', 'send_delay', 'stop_failure', 'clock_mismatch',
          'map_mismatch')


def setup(yaw=0.):
    clock = FakeClock()
    client = MockEduClient(clock)
    world = FlatWorld(clock, client, yaw)
    runtime = NavigationRuntime(EduSportSink(client), clock)
    runtime.update_map('mock-map', world.grid())
    return clock, client, world, runtime


def scenario(name, record=None):
    yaw = math.radians(float(name.split(':')[1])) if name.startswith('route:') else 0.
    clock, client, world, runtime = setup(yaw)
    goals = [(3., 0.), (0., 0.)] if name.startswith('route:') else [(3., 0.)]
    goal_index = 0
    frame = world.frame()
    assert runtime.set_goal(goals[0], frame, yaw=0.)
    runtime.arm(frame)
    event, resumed, steps = None, False, 0
    for step in range(5000):
        frame, manual = world.frame(), None
        if name in FAULTS and step == 20:
            if name == 'robot_dropout': frame = replace(frame, robot=replace(frame.robot, connected=False))
            elif name == 'pose_stale': frame = replace(frame, pose=replace(frame.pose, stamp=clock()-.26))
            elif name == 'unmatched_map': frame = replace(frame, pose=replace(frame.pose, matched=False))
            elif name == 'scan_dropout': frame = replace(frame, scan=None)
            elif name == 'invalid_scan': frame = replace(frame, scan=replace(frame.scan, ranges=(math.nan,)*120))
            elif name == 'partial_scan': frame = replace(frame, scan=replace(frame.scan, angles=frame.scan.angles[:30], ranges=frame.scan.ranges[:30]))
            elif name == 'obstacle':
                ranges = list(frame.scan.ranges); ranges[60] = .42
                frame = replace(frame, scan=replace(frame.scan, ranges=tuple(ranges)))
            elif name == 'operator_release': frame = replace(frame, robot=replace(frame.robot, enabled=False))
            elif name == 'emergency': frame = replace(frame, robot=replace(frame.robot, emergency=True))
            elif name == 'manual_takeover': manual = VelocityRequest(clock(), (0., 0., 0.))
            elif name == 'send_failure': client.move_code = -1
            elif name == 'send_exception': client.raise_move = True
            elif name == 'send_delay': client.delay = .16
            elif name == 'stop_failure':
                client.stop_code = -1
                frame = replace(frame, robot=replace(frame.robot, enabled=False))
            elif name == 'clock_mismatch': frame = replace(frame, clock_id='wall')
            elif name == 'map_mismatch': frame = replace(frame, pose=replace(frame.pose, map_id='different-map'))
        decision = runtime.step(frame, manual)
        if record:
            record.write(json.dumps(dict(scenario=name, step=step, time=clock(),
                                         pose=world.pose.tolist(), decision=asdict(decision)))+'\n')
        if name in FAULTS and step == 20:
            event = decision
            client.move_code, client.delay, client.raise_move = 0, 0., False
        if name in FAULTS and step > 20:
            resumed |= decision.source != 'stop'
        if name in FAULTS and step >= 30:
            break
        if name.startswith('route:'):
            if decision.reason == 'goal_reached':
                error = np.linalg.norm(world.pose[:2] - goals[goal_index])
                assert error <= .15, (name, error)
                goal_index += 1
                if goal_index == len(goals):
                    break
                assert runtime.set_goal(goals[goal_index], frame, yaw=0.)
                runtime.arm(frame)
            elif decision.source == 'stop':
                break
        world.advance()
        steps += 1
    if name in FAULTS:
        expected = {'robot_dropout':'robot_state_unavailable', 'pose_stale':'localization_unavailable',
                    'unmatched_map':'localization_unavailable', 'scan_dropout':'scan_unavailable',
                    'invalid_scan':'motion_not_clear', 'partial_scan':'motion_not_clear',
                    'obstacle':'motion_not_clear', 'operator_release':'operator_disabled',
                    'emergency':'emergency_latched', 'manual_takeover':'zero_request',
                    'send_failure':'transport_failure', 'send_exception':'transport_failure',
                    'send_delay':'transport_deadline_missed', 'stop_failure':'stop_not_acknowledged',
                    'clock_mismatch':'clock_domain_mismatch', 'map_mismatch':'localization_unavailable'}[name]
        passed = event is not None and event.reason == expected and not resumed
        # Missing ACK is a detected failure, not a claim that the mock plant stopped.
        passed &= (not event.transport_ok if name == 'stop_failure' else client.velocity == (0.,0.,0.))
    else:
        passed = goal_index == len(goals) and world.collisions == 0
    return dict(name=name, passed=bool(passed), simulated_seconds=clock(), steps=steps,
                goals_completed=goal_index, collisions=world.collisions,
                terminal_distance=float(np.linalg.norm(world.pose[:2]-goals[-1])),
                fault_reason=None if event is None else event.reason,
                resumed_without_rearm=resumed, physical_stop_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'trace.jsonl').open('w') as trace:
        results = [scenario(name, trace) for name in
                   ['route:0','route:-30','route:30','route:60','route:180']+list(FAULTS)]
    report = dict(backend='mock-edu-2d', hardware_verified=False, slam_verified=False,
                  passed=sum(r['passed'] for r in results), total=len(results), results=results)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
