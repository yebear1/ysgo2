# Go2 真 VSLAM 全链路导航回归

这套回归验证从视觉建图到重启定位、闭环导航和跟踪恢复的完整链路。导航控制只读取相机、IMU、RTAB-Map 位姿和 RTAB-Map 生成的占据栅格；MuJoCo 真值仅进入评分器，不会反馈给定位、规划或控制。

## 测试流程

`run_vslam_e2e_benchmark.sh` 自动执行两个相互独立的进程阶段：

1. 建图阶段在家庭环境中按传感器位姿探索，建立并保存 RTAB-Map 数据库。
2. 建图进程完全退出后，定位阶段以只读方式重新加载该数据库。
3. 依次导航到客厅、卧室、厨房和起点。
4. 到达卧室后遮挡 RGB 和深度图 2.5 秒，再主动转动搜索并测量重定位时间。
5. 汇总目标到达、家具碰撞、回环、漂移、返回误差和重定位结果。

入口使用单实例锁；重复启动会立即退出，避免多组 ROS 发布者污染相机、TF、数据库和测试报告。
RTAB-Map 在独立进程组中运行；退出时先等待数据库刷新，超时才升级终止信号，避免残留节点污染下一次回归。

建图路线和通过阈值位于 `deploy/deploy_mujoco/configs/vslam_e2e_benchmark.yaml`。单个探索目标连续 45 仿真秒没有明显接近，或视觉跟踪反复丢失超过预算时会明确失败，避免定位异常时无限转圈或靠短暂假恢复刷新超时。

## 运行

依赖 ROS 2 Jazzy、`rtabmap_ros`、RViz2、MuJoCo 和本项目 Python 环境。当前工作站可直接运行：

```bash
cd /home/user/下载/go2_rl_gym
./run_vslam_e2e_benchmark.sh
```

回归默认不启动 RViz，以减少渲染负载；MuJoCo 窗口仍会显示。普通交互式 VSLAM 使用：

```bash
./start_go2_vslam.sh --new-map
./start_go2_vslam.sh --localization
```

## 结果文件

- `maps/vslam_e2e_benchmark.rtabmap.db`：持久化 RTAB-Map 数据库。
- `reports/vslam_e2e/mapping.json`：建图阶段原始指标。
- `reports/vslam_e2e/localization.json`：重启定位阶段原始指标。
- `reports/vslam_e2e/report.json`：机器可读的总报告。
- `reports/vslam_e2e/report.md`：便于阅读的 PASS/FAIL 报告。
- `logs/rtabmap.log`：RTAB-Map 运行日志。

如果建图失败，脚本会跳过定位阶段并仍生成报告，退出码为非零。数据库和报告属于运行产物，已由 `.gitignore` 排除。

## 通过标准

- 客厅、卧室、厨房和起点全部到达。
- 家具碰撞为 0。
- 返回起点位置误差小于或等于 0.25 m。
- 最终航向误差小于或等于 10°。
- 人为遮挡导致跟踪丢失后 5 秒内恢复。
- 至少检测到一次回环。
- 保存数据库后重启仍能完成只读定位导航。

## 真实性边界

`VslamE2EBenchmark.update_control()` 不读取 MuJoCo 真值。真值只在 `observe()` 中计算位置和航向漂移、目标真实误差以及碰撞评分。评分器可在漂移超过安全界限时终止已经无效的测试，但不会校正 RTAB-Map 位姿。

这意味着报告失败是有效结果：它表明当前视觉里程计、回环或重定位尚未达到门槛，而不是由固定地图或真值位姿掩盖问题。

## 快速自检

```bash
/home/user/unitree_mujoco/.venv/bin/python -m py_compile \
  deploy/deploy_mujoco/vslam_e2e_benchmark.py \
  deploy/deploy_mujoco/summarize_vslam_e2e.py
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/test_vslam_e2e_benchmark.py
bash -n run_vslam_e2e_benchmark.sh start_go2_vslam.sh
```
