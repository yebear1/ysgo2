# Go2 EDU + Jetson TX1：第一阶段上机准备

当前范围：**只在开发电脑准备代码与离线测试，未部署、未连接、未驱动机器狗。**
机器人使用已有 EDU 底层运控；首阶段通过官方 Sport API 验证通信与低速指令。
本工具不替换步态，不发送 `rt/lowcmd`，不自动站立、不切换或释放运控模式。

## 1. 先确认 TX1 环境

TX1 的官方支持止于 JetPack 4 系列；JetPack 4.6.6 / L4T 32.7.6 的基础系统为
Ubuntu 18.04。这不是要求重刷机器：保留现有能运行 EDU 运控的系统，先采集实际版本。
官方来源：[JetPack 归档](https://developer.nvidia.com/embedded/jetpack-archive)、
[L4T 32.7.6](https://developer.nvidia.com/embedded/linux-tegra-r3276)。

以后进入部署阶段时，在 TX1 仓库根目录执行以下只读命令（本次未执行）：

```bash
python3 deploy/deploy_real/collect_tx1_info.py > tx1-info.json
```

它兼容 Python 3.6，仅读取系统、L4T、内存、网卡地址及已装 Python 包信息。
不连接机器人、不改网卡、不安装依赖。报告含本机 IP，按设备信息保存。

SDK 工具以 Python 3.8+ 为目标，首阶段无需 Torch、CUDA、ROS、MuJoCo。
若原系统仍是 Python 3.6，保留系统 Python；之后根据采集结果准备独立的 Python 3.8 环境，
不能覆盖 `/usr/bin/python3`，也不能直接安装开发电脑的 x86_64 wheel 到 TX1 的 aarch64 系统。
本次只在 x86_64 / Python 3.12 做离线测试及 3.8 语法检查，尚未验证 TX1 原生运行。

宇树 SDK 的安装应使用目标机兼容版本，记录来源提交并核对其 `setup.py` 中的依赖。
当前本地 SDK 声明 Python >=3.8、CycloneDDS 0.10.4；在线 README 的版本示例可能不同，
不要混装不同来源的 Python 绑定与原生库。SDK 导入报 `libddsc` 缺失时，按
[官方安装说明](https://github.com/unitreerobotics/unitree_sdk2_python#installation)
为该版本准备 CycloneDDS，并设置相应 `CYCLONEDDS_HOME`。具体安装命令待 TX1 信息确认后生成。

## 2. 离线预检（不建立 DDS 连接）

以下 `python` 指准备好的 Python 3.8+ 独立环境：

```bash
python deploy/deploy_real/go2_edu_bringup.py check
```

检查 Python、架构、网卡名和 SDK 动态库导入，失败退出码为 2。
`local_import_check_passed=true` 仅表示本地导入成功；`hardware_verified` 始终为 false。
此工具不会自动选择机器人网卡。当前开发电脑预检实际发现 CycloneDDS 动态库缺失，
这是开发环境结果，不代表 TX1 也缺同一依赖。不要把该项记录成通过。

## 3. 只读连接与本体传感器检查（留待部署阶段）

确认机器人接口权限、固件版本、网络接口及当前运控服务。准备独占测试环境，
退出其他可能发运动命令的应用或程序；保持已有官方运控，不按旧 RL 部署说明关闭它。
网卡名必须来自 TX1 的采集结果；下例先交互输入，避免把示例接口当成真实接口。

```bash
read -r -p '机器人网卡名: ' GO2_NET
python deploy/deploy_real/go2_edu_bringup.py listen --net "$GO2_NET" --seconds 10
python deploy/deploy_real/go2_edu_bringup.py listen --net "$GO2_NET" --seconds 10 --query-mode
```

普通 `listen` 只订阅 `rt/lowstate`；`--query-mode` 额外调用只读 `CheckMode` RPC。
输出 IMU 四元数/角速度/加速度、前 12 个电机位置/速度、足力、遥控器位域与摇杆。
先观察静止数据，再手动操作遥控器核对 R1/A/Select 与摇杆方向；此时不发送运动指令。
首次收包等待上限为 5 秒；后续新 tick 接收间隔超过 0.20 秒、数据无效或电机报丢帧则退出。
重复 tick 不刷新超时，tick 正常 uint32 回绕可继续，倒退则锁定失败并要求重新检查。
本阶段验证的是数据形状、有限数值和接收端活性；不替代传感器标定或传输延迟测量。

相机/雷达尚未提供型号，驱动配置不做假定。接导航前还需要实测：RGB/深度与相机内参、
有效深度单位、点云/扫描覆盖范围、IMU 坐标定义、时间戳来源、传感器到 `base_link` 外参。
已有本体 IMU/足力不等于已经具备 RGB-D 或避障距离传感器。

## 4. 单轴、短时低速测试（留待现场确认后执行）

前提：只读检查通过、操作者通过原有方式让机器人稳定站立、确认独立的停止方式可用，
并在开阔平地留足空间。现场先确认 R1+A 组合不会与该固件遥控器的其他功能冲突；
这些键只是软件使能输入，不是经认证的急停或遥控器失联检测。

读取 `--query-mode` 的成功返回并人工确认这是已有的官方运控模式，然后输入其 `name`。
工具仅核对名称，不自动选择模式。单次只测试一个轴，结束就退出：

```bash
read -r -p '已确认的官方运控 mode.name: ' GO2_MODE
python deploy/deploy_real/go2_edu_bringup.py pulse \
  --net "$GO2_NET" --expected-mode "$GO2_MODE" \
  --axis vx --speed 0.05 --seconds 1 --enable-motion
```

运行后先释放 R1 和 A，再持续按住 R1+A。必须先观察到释放态才接受使能；
15 秒未使能则退出。依次人工验证前后方向；之后再单独验证 `vy` 侧移、`yaw` 转向。
`vx/vy` 上限为 0.10 m/s，`yaw` 上限为 0.20 rad/s，单次最多 2 秒。
上述是初次试验的限制，不是宣称机器人硬件安全限值。首条指令保持 0.05 m/s、1 秒。
不要连续循环执行脚本或在这个阶段自动串联运动序列。

保护逻辑包括：摇杆居中、机身倾角不超过 20°、Select 终止、R1/A 任一释放终止、
状态异常终止、发送失败终止。使能等待后再次检查运控模式，避免等待期间模式变化。
尝试发送运动后，无论正常完成、Ctrl+C 或异常退出，都会尝试 3 次 `StopMove`；
没有收到成功回复会明确报错。`Move` 的发送成功不证明机器人执行成功，
`StopMove` 的服务回复也不证明物理速度已归零，现场仍须观察实际停止。
进程被 SIGKILL、断电、网络完全中断或 SDK 永久阻塞时，Python 的 finally 不能保证停机；
独立停止手段和底层超时行为必须现场验证。本工具不承诺硬实时停止时延。

## 5. 接导航前的明确边界

- 这条入口验证的是已有官方运控，未验证自定义 RL 步态。
- 旧 `deploy_real_go2.py` 配置指定的 `go2_moe_cts_137000_0.6365.pt` 当前不存在。
  不擅自替换为同目录另一个模型；需要后续单独确认模型来源、观测顺序、关节顺序与训练配置。
- 仿真家庭场景用的是 263 维特权观测教师策略，其中包含 MuJoCo 重建的接触力、高程等信息；
  它不能直接输入现有 45 维实机观察，也不能用填零代替缺失观测来声称可上机。
- 此次修正旧辅助函数的 `qd` 为 SDK 真正序列化的 `dq`，防止零力矩/阻尼指令残留速度字段；
  这并不意味着旧 RL 实机控制器已经具备完整的失联保护和实机验收。
- 公共导航核心、指令仲裁和 Mock EDU 故障测试已补齐，见
  [无实物阶段准备](GO2_OFFLINE_PREPARATION.md)。真实 VSLAM 传感器驱动、
  `use_sim_time=false`、时间转换、外参标定与物理停机仍待实机阶段验证。
- 不直接把开发电脑的 ROS 2 Jazzy 环境搬到 TX1。收到系统和传感器信息后，
  再决定 TX1 本机运行 SLAM，还是用外部计算机处理感知、TX1 保留控制通信；当前未做性能承诺。
- 仿真的 10/10 成功率不作为上述任一实机验收项的通过证据。

## 本地复核

```bash
python deploy/deploy_real/test_go2_edu_bringup.py
python deploy/navigation/verify_offline.py --output outputs/go2-offline-prep
```

这些测试使用假时钟、假客户端和结构化消息，不初始化 DDS，不与机器人连接。
验证限速/限时、持键拒绝自动使能、模式变化、状态超时、释放/Select/倾斜停止、发送异常、
停止回复失败、无效数据锁定和 `dq` 字段清零。它们不能替代 TX1、真实固件或物理停止测试。

官方接口依据：
[SportClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py)、
[MotionSwitcherClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py)、
[MotorCmd 字段](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/idl/unitree_go/msg/dds_/_MotorCmd_.py)。
