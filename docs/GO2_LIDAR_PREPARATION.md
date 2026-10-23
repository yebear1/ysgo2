# Go2 EDU 激光建图、保存地图定位与离线导航准备

本实现增加摄像头独立的激光路线，保留原有视觉 SLAM 和 EDU 低速测试入口。
当前交付是**工作站建图、只读采集、在线指令分析和离线验证**；不会连接运动 SDK、
发布 `/cmd_vel`、调用 `Move/StopMove`，也不会改变机器狗模式。没有实机部署结果。

## 已确认的官方接口

截至 2026-10-23，本轮依据以下固定版本资料准备，避免把不同产品的 SDK 当成同一接口：

- [Unitree ROS 2 README](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/README.md)
  明确演示 Go2 的 `/utlidar/cloud`，点云坐标系为 `utlidar_lidar`。
  `configs/go2_builtin.json` 仅将这两个值作为已确认的默认值。
- [L1 SDK](https://github.com/unitreerobotics/unilidar_sdk/blob/1bd7d95d8ab7ce7a22058d2bb07e39fd62612aa6/README.md)
  和 [L2 SDK](https://github.com/unitreerobotics/unilidar_sdk2/blob/0e3c51f512e6b8ff60b8c32f160b412cb48445c2/README.md)
  均提供点云/IMU 的开发入口，但属于对应雷达产品的驱动。
  不能由此推断它们可以直接替换 Go2 机载固件的数据服务。
- [Go2 产品页](https://www.unitree.com/cn/go2/)介绍内置雷达和 App 建图能力。
  商用 App 有建图功能，不等于本项目已经获得其保存地图、重定位状态、地图身份等开发接口。
  本实现没有使用未经核实的服务编号，也没有把官方里程计自动标为“保存地图已匹配”。

机器到位后核对 L1/L2、固件、点云字段、实际 TF 和 IMU 话题。先用已有机载输出，
本轮不启动另一个雷达驱动，不改雷达工作模式。启动 ROS 发现之前，由操作人员选择
正确的网络和 ROS domain；离线测试不需要连接机器狗。

## 架构与算法边界

1. `go2_lidar_mapping.launch.py` 启动 RTAB-Map 的 ICP 激光里程计与点云建图，或接受
   `external_odom_topic` 指定的外部里程计。没有图像订阅，不依赖摄像头。
2. `ros_capture.py` 按点云采样时刻查询安装 TF，解码 `PointCloud2`，截取相对机身的
   高度区间，保留每个角度区间最近的回波，并记录同步里程计和可选 IMU。
3. `SavedGrid` 保存带内容 SHA-256 的二维地图。旋转的地图原点、非法占据值和损坏
   的地图身份会被拒绝。文件已存在时不会覆盖。
4. `GridLocalizer` 在给定起始位置区域内搜索全部朝向，检查墙面距离、射线路径的
   已知空闲支持、候选差距以及连续三帧一致性。里程计只用于相对运动预测。
5. `AnalysisSession` 把匹配位姿接入现有 `NavigationRuntime`、路径规划、限速和扫掠
   检查。输出只在内存和分析记录中。实时分析与 JSONL 回放使用同一个会话实现。

这是平层室内的实验性二维保存地图匹配，不是通用三维重定位。启动需要提供地图坐标
中的大致位置：`search-radius` 是 X/Y 两个方向各自的半宽，最大 3 m，并非任意全图搜索。
姿态搜索不需要初始 yaw。对称环境或重复结构可能无法确认位置，此时保持未定位。
匹配分数不是经实机标定的概率或精度保证。

默认后端是 ICP，**没有宣称实现 LiDAR–IMU 紧耦合融合或点级运动去畸变**。
IMU 当前用于采集和后续质量复核。需要 FAST-LIO 等外部 LIO 时，先验证每点时间、
同步、安装外参和对应雷达适配，再用 `external_odom_topic` 接入其输出。
现有回放通过不证明此类外部后端已接通。

## 开发环境与 TX1

离线模块需要 Python 3.8+、NumPy、OpenCV。ROS 工具另需 ROS 2、`rclpy`、
`sensor_msgs_py`、`tf2_ros`，建图需 `rtabmap_odom`、`rtabmap_slam`。
本机验证使用 Ubuntu/ROS 2 Jazzy 的 x86_64 工作站；不是 TX1 原生运行结果。

先在开发电脑完成建图、匹配与录制，TX1 到位后按
[EDU/TX1 首阶段指南](GO2_EDU_TX1_BRINGUP.md)采集系统信息，再决定运行位置。
不要为了本流程直接重刷 TX1 或在旧 JetPack 上套用 Jazzy 安装命令。
需要实测 ARM 依赖、内存、点云带宽、持续温度和最坏处理延迟，再决定是否迁移计算。
本轮没有更换 EDU 原厂底层运控。

## 不接实物的验证命令

以下从仓库根目录运行。每次给激光矩阵、ROS smoke 使用一个新的输出目录：

```bash
python -m deploy.lidar.test_lidar
python -m deploy.lidar.run_offline --output outputs/lidar-planar-run1
python deploy/navigation/verify_offline.py --with-mujoco --output outputs/regression-run1
```

前两项是协议/故障测试和五种起始朝向的二维合成测距往返；真值只用于生成传感器、
评价与碰撞检查，不传给定位器。机器人运动仍是理想二维运动，没有步态、接触动力学。
矩阵只走 1 m 往返，不能代替三个房间或狭窄通道长时任务。

在已具备 ROS 2 Jazzy 和 RTAB-Map 的开发机上：

```bash
source /opt/ros/jazzy/setup.bash
python -m deploy.lidar.ros_smoke --output outputs/lidar-ros-run1 --domain 87
```

该命令限制本机 ROS 发现，并使用独立 domain，实际启动 ICP/建图/采集/导图节点。
合成三维点云和 IMU 驱动实际 ROS 程序，ICP 不接收真值里程计。
输出包括 RTAB-Map 数据库、地图、点云投影记录、在线只读分析、离线回放报告和日志。
它验证本机软件链路，不模拟 L1/L2 的扫描时序、真实噪声或真实机身。
测试期间不要在同一 domain 上运行其他仿真；不要把测试用的恒等安装 TF 用到实机。

## 实录前的数据确认

先只读检查已有 ROS 输出：

```bash
ros2 topic list -t
ros2 topic info /utlidar/cloud --verbose
ros2 topic echo /utlidar/cloud --once --no-arr
```

依据结果复制并修改 `deploy/lidar/configs/go2_builtin.json`，确认：

- `cloud_topic`、`cloud_frame`、`base_frame` 与实际发布者一致。
- 必须已有测量/标定得到的 `base_frame ← cloud_frame` TF。本程序不伪造安装坐标。
- `odom_topic` 为空时从 TF 读取 `odom_frame ← base_frame`；非空时读取标准
  `nav_msgs/Odometry`，检查 header/child frame，选择与点云时间相差不超过 50 ms 的样本。
- `imu_topic` 默认空，不能猜测机载 IMU 话题。配置后记录角速度、加速度、原始帧名
  和转换后的采样时间；不把未知安装方向的 IMU 数据当成机身姿态。
- `min_z=-0.27`、`max_z=0.50` 是针对示例机身高度的占障截取区间，单位 m，原点在
  `base_frame`，**不是离地高度**。实际机身高度、负载和安装参数必须测量后修改。
- 只支持低速平地。倾角超限拒收；高度截取没有实现坑洞检测，也可能漏掉截取区间以外
  的障碍。不能据此放行台阶、悬空边缘、桌面下方或任意地形。

点云解析保留大小端、字段偏移和行填充语义；NaN 点被剔除。
没有返回的角区间保持 0（未知），不会填最大量程假装空闲，故稀疏点云可能使导航分析
拒绝运动。需要先验证覆盖，不能通过放宽检查来掩盖传感器盲区。

## 工作站建图和保存地图

先确保只有一个所需坐标变换的发布者。默认 ICP 不发布 odom TF，以免覆盖已有机器人
TF；RTAB-Map 可使用同时间戳的里程计消息，可能提示没有对应 TF。
若在独立采集系统中由本 ICP 唯一提供 odom TF，才启用 `publish_odom_tf:=true`。
若已有可靠的激光惯性里程计，使用它的 `external_odom_topic` 和一致的 TF。

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch "$PWD/deploy/lidar/go2_lidar_mapping.launch.py" \
  database:="$PWD/outputs/site-a.db" \
  cloud_topic:=/utlidar/cloud base_frame:=base_link \
  ground_height:=-0.27
```

`ground_height` 是相对 base 的地面分类上界，必须与实测机身高度和采集配置一致。
该入口使用平面假设和按高度分类，生成二维导航栅格。`Grid/RayTracing` 开启，避免只
看到稀疏地面点而把整个可通行区域都保留为未知；没有观测到的区域仍保留未知。
新建图拒绝使用已经存在的数据库路径，绝不自动 `--delete_db_on_start`。
`mode:=localization` 可只读加载已有数据库，但 ICP 的局部收敛不提供任意朝向启动保证；
本项目的独立栅格定位器仍需通过上述全朝向搜索门控。

在建图过程中由操作者通过原厂控制手段移动机器狗；本程序没有遥控/自主运动入口。
建图输出足够完整后另开终端保存导航地图：

```bash
python -m deploy.lidar.ros_capture save-map \
  --topic /go2_lidar/map --output outputs/site-a.npz
```

`.db` 保存 RTAB-Map 数据，`.npz` 保存本导航器读取的栅格及地图身份，两者用途不同。
换图时必须创建新定位/导航会话，旧任务和旧使能不能跨地图继续运行。

## 只读采集、在线分析与回放

使用 ICP 的配置模板：

```bash
python -m deploy.lidar.ros_capture capture \
  --config deploy/lidar/configs/go2_workstation_icp.json \
  --output outputs/site-a-capture.jsonl --seconds 60
```

数据先等待对应 TF/里程计，队列有容量和数据年龄限制；迟到、重复、倒退、缺失、
覆盖不足等原因会保留在记录里。源时间与接收时间必须同步，源年龄不会被接收时刻覆盖。
ROS 时钟相对单调时钟发生跳变会锁定故障，需要重新建立采集会话。
当前实时采集使用系统 ROS 时钟；不要用加速 `/clock` 回放伪装实时流。

该 JSONL 保存导航使用的二维投影、里程计、可选 IMU 和点云字段说明，不是完整原始点云。
需要排查点级运动畸变时，另用 `ros2 bag record` 保存已确认的话题、`/tf` 和 `/tf_static`。
不要把投影 JSONL 当成可重新运行任意三维 SLAM 的原始数据。

已有 `.npz` 时可边采集边分析；目标只用于计算并记录指令，不发送给机器人：

```bash
python -m deploy.lidar.ros_capture capture \
  --config deploy/lidar/configs/go2_workstation_icp.json \
  --output outputs/site-a-live.jsonl --seconds 60 \
  --map outputs/site-a.npz --search-center 0 0 --search-radius 1 \
  --evaluate-goal 1 0 --analysis-output outputs/site-a-analysis.jsonl
```

对应的离线回放：

```bash
python -m deploy.lidar.cli replay \
  --recording outputs/site-a-live.jsonl --map outputs/site-a.npz \
  --search-center 0 0 --search-radius 1 \
  --evaluate-goal 1 0 --output outputs/site-a-replay.json
```

没有 `--evaluate-goal` 时只评价定位。位置区域和目标均是保存地图坐标，不是开机里程计
的 `(0,0)`。场地起点不明确时应先通过地图人工确认，不能把真值或任意初始 TF 当成匹配。
图已加载、ICP/LIO 正常但没有保存地图匹配时，输出保持停止；候选歧义、数据丢失、
不同步、定位跳变也拒绝运动。运行中故障后即使定位恢复，原任务不自动重新使能。
回放还计入本机匹配计算耗时，处理超时不能靠重写传感器时间掩盖。
当前录制未提供测量机身速度时，终点控制分析使用零速度假设；不据此预测实机停车过程。

## 验证结论应怎样表述

区分三种证据：单元/故障测试、二维理想闭环、实际 ROS 节点处理合成三维测量。
不能把三者相加形成“实机成功率”。实际雷达的时间字段、扫描畸变、网络抖动、地面分类、
真实定位误差、停止距离与 TX1 长时性能仍需实录和实机验证。

[既有视觉回归](../regressions/vslam/20260926-startup-terminal/README.md)属于另一条已保存的
仿真证据，本次激光结果不能替代它，也不能直接证明此前狭窄通道和返程问题已在实机解决。
