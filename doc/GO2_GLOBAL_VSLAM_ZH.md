# Go2 全局 RGB-D VSLAM

这套管线使用 RTAB-Map 完成视觉里程计、关键帧管理、外观回环检测和位姿图优化。MuJoCo 进程只发布 RGB-D 和本体 IMU 测量以及传感器标定；全局位姿和全局地图不读取仿真世界坐标或真值位姿。IMU 姿态由机身坐标系的陀螺仪积分得到，用于在转向和低纹理画面中约束视觉里程计。

## 数据流

```text
MuJoCo RGB-D 相机 + 本体 IMU
  -> /go2/camera/* + /go2/imu/data
  -> RTAB-Map RGB-D visual odometry
  -> odom -> base_link
  -> RTAB-Map loop closure + pose graph
  -> map -> odom
  -> /map
  -> GoalNavigator A* + RL 局部避障/步态
```

## 建图

新电脑先安装依赖：

```bash
cd /home/user/下载/go2_rl_gym
./install_vslam_deps.sh
```

首次创建一张全新地图：

```bash
cd /home/user/下载/go2_rl_gym
./start_go2_vslam.sh --new-map
```

普通启动会继续使用 `maps/go2_home.rtabmap.db`，不会删除已有关键帧和回环图：

```bash
./start_go2_vslam.sh
```

沿通道和房间缓慢走一圈并回到看过的位置。RViz 显示 `/map`，TF 树应为：

```text
map -> odom -> base_link -> go2_camera_optical_frame
```

`/go2/vslam/status` 中的 `loops` 是已接受的回环次数。

## 保存

RTAB-Map 数据库在运行期间持续写入 `maps/go2_home.rtabmap.db`。需要同时导出 Nav2 兼容的二维导航地图时运行：

```bash
./save_go2_map.sh go2_home
```

输出：

- `maps/go2_home/map.yaml`
- `maps/go2_home/map.pgm`
- `maps/go2_home/go2_home.rtabmap.db`

## 加载地图并定位导航

```bash
./start_go2_vslam.sh --localization
```

定位模式加载数据库但不继续扩展地图。RTAB-Map 输出经过回环图优化的 `map -> base_link` 位姿；全局 A* 只读取 `/map`，未知区域视为不可通行。在 RViz 中选择 **2D Goal Pose** 并点击地图，或按数字键选择预设目标；RL 策略继续负责行走，3D LiDAR 继续负责台阶、近距离避障和防跌落。

如果需要指定另一个数据库：

```bash
./start_go2_vslam.sh --localization --database=/绝对路径/地图.rtabmap.db
```

## 诊断

```bash
ros2 topic echo /go2/vslam/status
ros2 topic hz /rtabmap/odom
ros2 topic hz /map
ros2 run tf2_ros tf2_echo map base_link
```

RTAB-Map 详细日志位于 `logs/rtabmap.log`。
