# Go2 家庭导航自动回归测试

这套测试用于防止“修好一个位置、另一个窄通道又退化”。无窗口路线测试使用与交互仿真相同的 Go2 碰撞模型、平面/高程雷达、高层 PPO、低层 perceptive RL 和安全状态机。

## 一键运行

```bash
cd /home/user/下载/go2_rl_gym
./run_navigation_benchmark.sh
```

只跑指定路线：

```bash
./run_navigation_benchmark.sh --only tv_console_pass armchair_narrow_passage
```

输出：

- `reports/navigation_benchmark_latest.json`：机器可读指标
- `reports/navigation_benchmark_latest.md`：便于查看的报告
- `reports/stairs_benchmark_latest.txt`：完整楼梯物理测试记录

## 当前固定路线

- `tv_console_pass`：电视柜前方直行
- `armchair_narrow_passage`：扶手椅与隔墙之间的窄通道
- `north_wall_corner_escape`：贴近墙角起步后自主脱困
- `multiroom_bedroom`：绕过家具和隔墙的跨房间导航

每条路线记录是否到达、仿真耗时、实际/规划路径长度、家具接触次数、最大接触力、阻塞时间、跌倒和卡死状态。接触统计排除地板与客厅地毯。

总入口还会运行雷达、窄通道对称控制、墙体避障、悬崖检测、VSLAM 与仿真真值隔离等快速测试，并自动跑完整楼梯物理测试。楼梯也可单独运行：

```bash
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/test_stairs_headless.py \
  --speed 0.55 --duration 12
```

路线和阈值保存在 `deploy/deploy_mujoco/configs/home_benchmark.yaml`。修改导航、碰撞体、RL 策略或地图参数后，应先运行本套件再接受新版。

`benchmarks/home_navigation_baseline.json` 保存当前策略的固定基线。报告同时区分两件事：严格质量门是否全部达标，以及结果是否比保存基线退化。已有缺陷不会被偷偷标成通过，但后续修改也不能让原本通过的路线倒退。
