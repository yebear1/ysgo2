#!/usr/bin/env python3
"""Read-only platform inventory; works on the stock Python 3.6 as well.
No network connections, DDS initialization, installs, or system changes.
"""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip().strip('\x00')
    except OSError:
        return None


def command(args):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=5)
        return {'exit_code': result.returncode, 'output': result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def collect():
    return {
        'machine': platform.machine(), 'kernel': platform.release(),
        'python': sys.version, 'python_executable': sys.executable,
        'device_model': read('/proc/device-tree/model'),
        'os_release': read('/etc/os-release'),
        'l4t_release': read('/etc/nv_tegra_release'),
        'memory': command(['free', '-m']),
        'interfaces': command(['ip', '-brief', 'address']),
        'jetson_packages': command(['dpkg-query', '-W', 'nvidia-jetpack', 'nvidia-l4t-core']),
        'python_packages': command([sys.executable, '-m', 'pip', 'show', 'unitree_sdk2py', 'cyclonedds', 'numpy']),
        'ros_distro_environment': os.environ.get('ROS_DISTRO'),
        'robot_connection_attempted': False,
    }


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
