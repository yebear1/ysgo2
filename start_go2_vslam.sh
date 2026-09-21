#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

mkdir -p /home/user/.ros

rviz2 -d deploy/deploy_mujoco/configs/go2_vslam.rviz &
rviz_pid=$!

cleanup() {
  kill "$rviz_pid" 2>/dev/null || true
  wait "$rviz_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/deploy_go2.py \
  --config go2_home.yaml --ros2-vslam "$@"
