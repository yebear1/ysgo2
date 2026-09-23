#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

mode="mapping"
new_map=0
database="$PWD/maps/go2_home.rtabmap.db"
sim_args=()
for argument in "$@"; do
  case "$argument" in
    --localization)
      mode="localization"
      ;;
    --new-map)
      new_map=1
      ;;
    --database=*)
      database="${argument#--database=}"
      ;;
    *)
      sim_args+=("$argument")
      ;;
  esac
done

if ! ros2 pkg prefix rtabmap_launch >/dev/null 2>&1; then
  echo "RTAB-Map is not installed. Install ros-jazzy-rtabmap-ros first." >&2
  exit 1
fi
if [[ "$mode" == "localization" && ! -f "$database" ]]; then
  echo "Localization database not found: $database" >&2
  exit 1
fi

mkdir -p "$(dirname "$database")" logs /home/user/.ros
if [[ "$new_map" == 1 && -f "$database" ]]; then
  backup="${database}.bak.$(date +%Y%m%d-%H%M%S)"
  mv "$database" "$backup"
  echo "Previous SLAM database preserved as: $backup"
fi

localization=false
memory_args="--Mem/IncrementalMemory true"
if [[ "$mode" == "localization" ]]; then
  localization=true
  memory_args="--Mem/IncrementalMemory false \
--Mem/InitWMWithAllNodes true \
--Mem/LocalizationReadOnly true \
--RGBD/StartAtOrigin true"
fi

rtabmap_args="--Reg/Force3DoF true \
--RGBD/CreateOccupancyGrid true \
--RGBD/OptimizeFromGraphEnd false \
--RGBD/OptimizeMaxError 3.0 \
--RGBD/ProximityBySpace true \
--RGBD/ProximityByTime true \
--RGBD/LocalRadius 4.0 \
--Vis/MinInliers 6 \
--Vis/MaxDepth 8.0 \
--Vis/MaxFeatures 2000 \
--Vis/FeatureType 1 \
--Vis/GridRows 2 \
--Vis/GridCols 3 \
--Kp/DetectorStrategy 1 \
--Kp/MaxFeatures 1500 \
--Mem/UseOdomFeatures false \
--RGBD/LoopClosureReextractFeatures true \
--RGBD/LinearUpdate 0.08 \
--RGBD/AngularUpdate 0.08 \
--Grid/FromDepth true \
--Grid/3D false \
--Grid/CellSize 0.05 \
--Grid/RangeMax 5.0 \
--Grid/DepthDecimation 1 \
--Grid/RayTracing true \
--Grid/NormalsSegmentation false \
--Grid/NoiseFilteringRadius 0.10 \
--Grid/NoiseFilteringMinNeighbors 2 \
--Grid/MinObstacleHeight -0.18 \
--Grid/MaxObstacleHeight 1.50 \
--Grid/MaxGroundHeight -0.18 \
--Grid/MinGroundHeight -0.40 \
$memory_args"

# These parameters belong only to the visual-odometry node. Passing them in
# the common RTAB-Map arguments makes the SLAM node reject undeclared Odom/*
# parameters on ROS 2 Jazzy.
odom_args="--Odom/ResetCountdown 3 \
--Odom/Strategy 1 \
--Odom/VisKeyFrameThr 15 \
--Odom/GuessMotion false \
--Vis/CorType 1 \
--Vis/BundleAdjustment 0 \
--Vis/FeatureType 8 \
--OdomF2M/BundleAdjustment 0"

rtabmap_log="$PWD/logs/rtabmap.log"
ros2 launch rtabmap_launch rtabmap.launch.py \
  namespace:=rtabmap \
  database_path:="$database" \
  localization:="$localization" \
  frame_id:=base_link \
  map_frame_id:=map \
  map_topic:=/map \
  vo_frame_id:=odom \
  publish_tf_odom:=true \
  publish_tf_map:=true \
  visual_odometry:=true \
  rgbd_sync:=true \
  approx_rgbd_sync:=false \
  approx_sync:=false \
  qos:=2 \
  rgb_topic:=/go2/camera/color/image_raw \
  depth_topic:=/go2/camera/depth/image_raw \
  camera_info_topic:=/go2/camera/color/camera_info \
  imu_topic:=/go2/imu/data \
  wait_imu_to_init:=true \
  always_check_imu_tf:=false \
  rviz:=false \
  rtabmap_viz:=false \
  args:="$rtabmap_args" \
  odom_args:="$odom_args" \
  >"$rtabmap_log" 2>&1 &
rtabmap_pid=$!

rviz2 -d deploy/deploy_mujoco/configs/go2_vslam.rviz &
rviz_pid=$!

cleanup() {
  kill -INT "$rtabmap_pid" 2>/dev/null || true
  kill "$rviz_pid" 2>/dev/null || true
  wait "$rtabmap_pid" 2>/dev/null || true
  wait "$rviz_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "RTAB-Map mode: $mode"
echo "Persistent database: $database"
echo "RTAB-Map log: $rtabmap_log"
sleep 2
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/deploy_go2.py \
  --config go2_home.yaml --ros2-vslam "${sim_args[@]}"
