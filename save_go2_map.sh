#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
set -u

name="${1:-go2_home}"
output_dir="$PWD/maps/$name"
database="${2:-${GO2_RTABMAP_DB:-$PWD/maps/go2_home.rtabmap.db}}"
mkdir -p "$output_dir"

if ros2 service list | rg -x '/rtabmap/backup' >/dev/null; then
  ros2 service call /rtabmap/backup std_srvs/srv/Empty '{}' >/dev/null
fi

ros2 run nav2_map_server map_saver_cli \
  -f "$output_dir/map" \
  --ros-args \
  -p map_subscribe_transient_local:=true \
  -p save_map_timeout:=15.0

if [[ -f "$database" ]]; then
  cp -a "$database" "$output_dir/$name.rtabmap.db"
fi

echo "Navigation map: $output_dir/map.yaml"
echo "Loop-closure database: $output_dir/$name.rtabmap.db"
