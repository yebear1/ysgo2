# Go2 VSLAM 全链路导航回归报告

- 总结果：通过
- 目标：4/4
- 家具碰撞：0
- 回环：50
- 建图进度：14/14
- 建图失败原因：无
- 定位失败原因：无
- 回到起点误差：0.133 m
- 最终航向误差：1.32°
- 遮挡后重定位：0.140 s
- 全程最大定位漂移：0.158 m

| 检查 | 结果 |
|---|---:|
| mapping_finished | PASS |
| database_saved | PASS |
| localization_finished | PASS |
| all_goals_reached | PASS |
| zero_furniture_collisions | PASS |
| return_position | PASS |
| return_yaw | PASS |
| tracking_loss_observed | PASS |
| relocalized | PASS |
| recovery_time | PASS |
| loop_closure | PASS |

全局导航使用 RTAB-Map 位姿和保存的占据栅格，局部避障使用模拟距离扫描。MuJoCo 真值位姿只用于评分，不校正导航；底层 RL 保留模拟本体感知和地形观测。
