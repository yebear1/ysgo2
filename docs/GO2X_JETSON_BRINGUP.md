# Go2 X + 外接 Jetson：内置雷达接入与离线建图

Go2 X 保留原厂运控。外接 NVIDIA Jetson 运行现有 `deploy/lidar` 的建图和定位，原有手机遥控先不改。本步骤只验证 ROS 2 点云入口，不发 `/cmd_vel`、不调用 `Move`，也不修改机器狗模式。不要把 EDU 运动适配器或现成 TF 参数直接套到 X 上。

## 当前进展（2026-11-22）

- 在 Jetson Orin NX Super（Ubuntu 24.04 / ROS 2 Jazzy）上通过 Cyclone DDS 收到 Go2 X 真机数据。实测 `/utlidar/cloud_base` 约 15.5 Hz，坐标系为 `base_link`；`/utlidar/robot_odom` 约 149 Hz，坐标系为 `odom → base_link`。这些是本机本次观测，不是固件通用保证。
- 短录包的 171 帧点云均与原厂里程计按源时间配对；最大时间差约 26 ms。隔离回放成功生成 RTAB-Map 数据库和 5 cm 分辨率的二维诊断栅格。该结果只验证数据链路，不验证地图覆盖或可通行性；原始录包和地图不随仓库发布。
- 实时定位未通过：实测狗端传感器时间与 Jetson 相差约 404 天，合成数据的板上实时定位测试也出现采集超时。没有修改狗端时钟、放宽超时保护或发送运动指令。下一步使用现有遥控在平层场地采集移动录包，验证拼图质量，再独立验证实时对时与重定位。

## 1. 在 Jetson 确认系统与话题

先确认原有树莓派程序如何连接 Go2 X：网口或 Wi-Fi、DDS 域或网桥，以及它订阅的究竟是机载雷达、状态还是手机控制话题。若使用 ROS 2，在 Jetson 使用同一条已知可用的发现路径，加载**本机已安装**的 ROS 2 环境：

```bash
tr -d '\0' </proc/device-tree/model; echo
cat /etc/nv_tegra_release 2>/dev/null
. /etc/os-release; echo "$PRETTY_NAME"
echo "ROS_DISTRO=${ROS_DISTRO:-未加载} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
ip -br addr
ros2 topic list -t
```

树莓派的 `go2-web-controller` 有 ROS 2 `sport` 和直连 DDS 的 `sdk_sport` 两种高层控制后端。迁移时核对树莓派实际使用的启动方式；代码里存在适配器不代表该适配器已在当前固件验证。`rt/lowstate` 等状态订阅不是雷达输入。若需要迁移遥控，在 Jetson 重新创建虚拟环境，不直接复用 U 盘里的树莓派虚拟环境或离线依赖包。此阶段也不启动 `dds_policy` 或发布 `rt/lowcmd`。

## 2. 站稳采集，不发运动指令

在 `ysgo2` 仓库根目录执行。`/utlidar/cloud` 是候选地址；如上一步发现别的原始 `sensor_msgs/msg/PointCloud2` 话题，改用其实际名称。

```bash
python3 -m deploy.lidar.go2x_probe --cloud-topic /utlidar/cloud --seconds 10
```

- `cloud_metadata_ok`：收到至少两帧；坐标系一致，源时间戳递增，点云结构含 `x/y/z`。这不保证地图可用，也未校验点值、雷达安装外参或时钟同步。
- `check_failed` 且 `received=0`：检查 JSON 里的 `matched_publishers` 和 `discovered_sensor_topics`，再用 `ros2 topic info <实际话题> --verbose` 检查发布端。若原遥控走的不是 ROS 2，先确认是否已有把雷达流转为 ROS 2 的网桥；不假定切网或换域就能收包。
- `last_header_age_s` 只是以 Jetson ROS 时钟减消息源时间得到的值。不同的时钟基准下它不是网络延迟。进入建图和同步采集前需要核对时间基准，不能用接收时间直接掩盖异常源时间戳。

## 3. 验收后再接建图与运动

先确认点云是可用于配准的局部扫描，量测 `base_frame` 到雷达 `frame_id` 的安装变换，并确认 TF 发布者唯一。若没有已验证可用的里程计，可用仓库现有 RTAB-Map ICP 工作站入口，参见 [激光建图准备](GO2_LIDAR_PREPARATION.md)。由操作员用现有遥控走一圈建图，再验证保存地图定位和只读规划。最后才接运动接口：复用已验证的遥控发送层，保证手动/自动互斥，并确认遥控可接管、通信或定位失效时能停止、恢复后不能自动开始移动。

## 4. 用真机录包做隔离建图

若 `/utlidar/cloud_base` 的消息坐标系确为 `base_link`，且 `/utlidar/robot_odom` 为 `odom → base_link`，可用短录包验证地图生成，不需要猜测雷达外参。`go2x_replay_bag` 只保留这两个话题，保留原始消息与头时间戳；按源时间配对并另写回放包，不修改输入。录包时确认原始数据来自独立的只读采集窗口。

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=96 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
unset CYCLONEDDS_URI ROS_LOCALHOST_ONLY ROS_STATIC_PEERS ROS_DISCOVERY_SERVER
python3 -m deploy.lidar.go2x_replay_bag /absolute/path/to/input_bag "$PWD/outputs/replay-bag"
ros2 launch "$PWD/deploy/lidar/go2_lidar_mapping.launch.py" \
  database:="$PWD/outputs/replay-diagnostic.db" use_sim_time:=true \
  base_frame:=base_link odom_frame:=odom cloud_topic:=/go2x_replay/cloud_base \
  external_odom_topic:=/go2x_replay/odom external_odom_tf_sync:=false
```

另开两个终端，加载同一环境（包括隔离域设置），先执行导出命令，随后播放：

```bash
python3 -m deploy.lidar.ros_capture save-map --output "$PWD/outputs/replay-diagnostic.npz" --seconds 30
ros2 bag play "$PWD/outputs/replay-bag" --clock 100 --disable-keyboard-controls --delay 2
```

这里的 `external_odom_tf_sync:=false` 只用于没有里程计 TF 的诊断录包：RTAB-Map 使用时间差小于 50 ms 的里程计消息，不做基于 TF 的扫描运动补偿。导出的是第一张有效栅格，不代表完成了整个场地建图。地面高度仍用默认诊断值，不可用来验收可通行性。这个回放流程不校准狗和 Jetson 的实时钟，也不更改实时定位的超时保护。
