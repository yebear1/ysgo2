# Go2 高层导航强化学习

## 当前架构

这套系统现在有两层强化学习控制：

1. 高层 PPO 读取目标方向、机体速度、上一条指令和 24 束平面雷达，输出 `Vx/Vy/Wz`。
2. 原有 Go2 低层 rough/CTS 策略读取速度指令和高程感知，输出 12 个关节动作。

RTAB-Map VSLAM 提供全局位姿和占据栅格，A* 提供全局路径，高层 PPO 负责逐路点局部运动，`TerrainNavigator` 对墙、楼梯和跌落保留最终安全否决权。

这不是运行时在线试错。当前策略是在随机室内二维环境里离线 PPO 训练后，以 TorchScript 模型部署；仿真运行时只推理，不会修改权重。

## 文件

- 训练器：`deploy/deploy_mujoco/train_high_level_nav.py`
- 运行时适配器：`deploy/deploy_mujoco/high_level_nav_policy.py`
- 策略模型：`deploy/pre_train/go2/high_level_nav.pt`
- 配置：`deploy/deploy_mujoco/configs/go2_home.yaml`
- 自检：`deploy/deploy_mujoco/test_high_level_navigation.py`

## 训练

```bash
cd /home/user/下载/go2_rl_gym
/home/user/unitree_mujoco/.venv/bin/python \
  deploy/deploy_mujoco/train_high_level_nav.py \
  --num-envs 128 --horizon 32 --iterations 100 --epochs 3 \
  --minibatch-size 1024 \
  --output deploy/pre_train/go2/high_level_nav.pt
```

这次训练共经历 12,282 个回合；训练末期成功率约 80.1%。使用不同随机种子的 512 个并行环境做确定性评估，共 9,165 个回合，成功率约 92.97%，碰撞率约 6.08%。二维训练指标不能替代 MuJoCo 三维验证。

## 启动

```bash
cd /home/user/下载/go2_rl_gym
./start_go2_vslam.sh --localization
```

导航时终端出现 `NAV-RL` 表示高层 PPO 正在生成速度指令。直行速度保持配置值；急转时会降低平移速度，以避免长机身和后腿在家具旁发生转弯扫碰。这个约束不膨胀障碍物，也不会缩窄直行通道。

## 下一阶段

若要让策略继续进化，下一步应把随机训练世界升级为与 Go2 尺寸一致的矩形机身、动态障碍物和真实传感器噪声，并建立固定的 MuJoCo 家庭场景回归路线。在线学习应在独立训练实例中完成，不应让正在导航的策略直接在线改权重。
