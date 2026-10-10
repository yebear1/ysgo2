#!/usr/bin/env python3
"""Run the hardware-free preparation suite and retain logs + source hashes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-mujoco', action='store_true',
                        help='Also run existing geometry/VSLAM/PPO regressions; needs their dev dependencies')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = ['deploy/deploy_real/test_go2_edu_bringup.py', 'deploy/navigation/test_navigation.py']
    if args.with_mujoco:
        scripts += ['deploy/deploy_mujoco/'+name for name in (
            'test_vslam_navigation.py', 'test_motion_sweep.py', 'test_vslam_e2e_benchmark.py',
            'test_ros2_vslam_bridge.py', 'test_terrain_navigator.py', 'test_navigation_geometry.py',
            'test_high_level_navigation.py')]
    commands = [[sys.executable, s] for s in scripts]
    commands.append([sys.executable, 'deploy/navigation/run_offline.py', '--output', str(output/'mock')])
    results = []
    for command in commands:
        name = Path(command[1]).stem
        print('Running '+name, flush=True)
        try:
            run = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, timeout=180)
            code, log = run.returncode, run.stdout
        except subprocess.TimeoutExpired as exc:
            code, log = 124, 'Offline test timeout: '+str(exc)
        (output/(name+'.log')).write_text(log)
        results.append(dict(name=name, exit_code=code))
    sources = list((ROOT/'deploy/navigation').glob('*.py'))
    sources += [ROOT/'deploy/deploy_real'/name for name in (
        'go2_edu_bringup.py','collect_tx1_info.py','test_go2_edu_bringup.py','common/command_helper.py')]
    sources += [ROOT/'deploy/deploy_mujoco'/name for name in (
        'shared_navigation.py','deploy_go2.py','goal_navigator.py','terrain_navigator.py',
        'high_level_nav_policy.py','vslam_e2e_benchmark.py','ros2_vslam_bridge.py','lidar_heightmap.py')]
    syntax_error = None
    try:
        for path in sources:
            ast.parse(path.read_text(), filename=str(path), feature_version=(3,8))
    except (SyntaxError, ValueError) as exc:
        syntax_error = str(exc)
    report = dict(passed=all(r['exit_code'] == 0 for r in results) and syntax_error is None,
                  hardware_verified=False, tx1_native_verified=False,
                  python=platform.python_version(), architecture=platform.machine(),
                  python38_syntax_error=syntax_error, checks=results,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS' if report['passed'] else 'FAIL')
    print(str(output/'verification.json'))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
