#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
set -u
# Protect direct launches too (the end-to-end wrapper has its own lock).
exec 8>"${XDG_RUNTIME_DIR:-/tmp}/go2_vslam_ros_${UID}_${ROS_DOMAIN_ID:-0}.lock"
if ! flock -n 8; then
  echo "A Go2 VSLAM instance already owns ROS domain ${ROS_DOMAIN_ID:-0}." >&2
  exit 3
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

mode="mapping"
new_map=0
database="$PWD/maps/go2_home.rtabmap.db"
sim_args=()
use_rviz=1
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
    --no-rviz)
      use_rviz=0
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
  # The startup heading is unknown. Search candidates across the full circle;
  # feature inlier and graph-error checks still decide whether a match is valid.
  memory_args="--Mem/IncrementalMemory false \
--Mem/InitWMWithAllNodes true \
--Mem/LocalizationReadOnly true \
--RGBD/LocalizationPriorError 0.03 \
--RGBD/ProximityMaxPaths 0 \
--RGBD/ProximityAngle 180 \
--RGBD/StartAtOrigin true"
fi

rtabmap_args="--Reg/Force3DoF false \
--Optimizer/GravitySigma 0.05 \
--Optimizer/Robust true \
--RGBD/CreateOccupancyGrid true \
--RGBD/OptimizeFromGraphEnd false \
--RGBD/OptimizeMaxError 3.0 \
--RGBD/ProximityBySpace true \
--RGBD/ProximityByTime false \
--RGBD/LocalRadius 4.0 \
--Vis/MinInliers 15 \
--Vis/EstimationType 0 \
--Vis/InlierDistance 0.04 \
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
--Grid/Sensor 1 \
--Grid/3D false \
--Grid/CellSize 0.05 \
--Grid/RangeMax 5.0 \
--Grid/DepthDecimation 1 \
--Grid/RayTracing true \
--Grid/NormalsSegmentation true \
--Grid/MapFrameProjection false \
--Grid/MaxGroundAngle 20 \
--Grid/NormalK 10 \
--Grid/FlatObstacleDetected false \
--Grid/MinClusterSize 3 \
--Grid/NoiseFilteringRadius 0.10 \
--Grid/NoiseFilteringMinNeighbors 2 \
--Grid/MaxObstacleHeight 1.50 \
--Grid/MaxGroundHeight 0 \
--Grid/MinGroundHeight 0 \
$memory_args"

# These parameters belong only to the visual-odometry node. Passing them in
# the common RTAB-Map arguments makes the SLAM node reject undeclared Odom/*
# parameters on ROS 2 Jazzy.
# Do not silently re-origin odometry after a few bad frames.  The benchmark
# actively scans for the previous local map; if that fails it reports a loss
# instead of continuing in a shifted coordinate frame.
odom_args="--Reg/Force3DoF false \
--Odom/ResetCountdown 0 \
--Odom/Strategy 0 \
--Odom/VisKeyFrameThr 80 \
--Odom/GuessMotion false \
--Vis/CorType 0 \
--Vis/MinInliers 15 \
--Vis/PnPMaxVariance 0.02 \
--Vis/BundleAdjustment 0 \
--Vis/FeatureType 8 \
--OdomF2M/BundleAdjustment 1 \
--OdomF2M/MaxSize 3000"

rtabmap_log="${GO2_RTABMAP_LOG:-$PWD/logs/rtabmap.log}"
setsid ros2 launch rtabmap_launch rtabmap.launch.py \
  use_sim_time:=true \
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
  rgbd_sync:=false \
  subscribe_rgbd:=true \
  depth:=false \
  rgbd_topic:=/go2/camera/rgbd_image \
  approx_rgbd_sync:=false \
  approx_sync:=false \
  qos:=1 \
  qos_imu:=2 \
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

rviz_pid=""
if [[ "$use_rviz" == 1 ]]; then
  rviz2 -d deploy/deploy_mujoco/configs/go2_vslam.rviz --ros-args -p use_sim_time:=true &
  rviz_pid=$!
fi

cleanup() {
  kill -INT -- "-$rtabmap_pid" 2>/dev/null || true
  if [[ -n "$rviz_pid" ]]; then
    kill "$rviz_pid" 2>/dev/null || true
  fi
  # ros2 launch can wait forever when an RTAB-Map worker is stuck saving or
  # shutting down.  Give the process group time to flush the database, then
  # escalate so benchmark runs cannot leave publishers behind.
  for _ in {1..20}; do
    if ! kill -0 "$rtabmap_pid" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if kill -0 "$rtabmap_pid" 2>/dev/null; then
    kill -TERM -- "-$rtabmap_pid" 2>/dev/null || true
    for _ in {1..10}; do
      if ! kill -0 "$rtabmap_pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
  fi
  if kill -0 "$rtabmap_pid" 2>/dev/null; then
    kill -KILL -- "-$rtabmap_pid" 2>/dev/null || true
  fi
  wait "$rtabmap_pid" 2>/dev/null || true
  if [[ -n "$rviz_pid" ]]; then
    wait "$rviz_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "RTAB-Map mode: $mode"
echo "Persistent database: $database"
echo "RTAB-Map log: $rtabmap_log"
sleep 2
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/deploy_go2.py \
  --config go2_home.yaml --ros2-vslam "${sim_args[@]}"
