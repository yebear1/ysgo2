#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy is required at /opt/ros/jazzy." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  ros-jazzy-rtabmap-launch \
  ros-jazzy-nav2-map-server

echo "RTAB-Map RGB-D SLAM and map saving dependencies are installed."
