# VSLAM 批量回归

状态：完成；已完成 10/10。
成功：4/10（所有预定样本作分母，未完成不算成功）。
已知接触事件：0；卡住事件：1。
缺少接触/卡住指标的运行：0/0。

卡住定义：导航激活且视觉有效时，至少 10 仿真秒位移不足 0.15 m、航向变化不足 20°；遮挡/失定位时间单独排除。
有意转身不计为卡住；长时间低效绕行仍可能超时，但不一定符合此静止判据。

| 运行 | 顺序（最后均回起点） | 出发朝向 | 结果 | 碰撞 | 卡住 | 仿真秒 | 墙钟秒 | 原因 |
|---|---|---:|---|---:|---:|---:|---:|---|
| order_01 | living_room → bedroom → kitchen | 0° | PASS | 0 | 1 | 142.88 | 287.62 |  |
| order_02 | living_room → kitchen → bedroom | 0° | PASS | 0 | 0 | 123.82 | 249.65 |  |
| order_03 | bedroom → living_room → kitchen | 0° | FAIL | 0 | 0 | 150.02 | 301.75 | timeout |
| order_04 | bedroom → kitchen → living_room | 0° | PASS | 0 | 0 | 112.36 | 228.60 |  |
| order_05 | kitchen → living_room → bedroom | 0° | FAIL | 0 | 0 | 150.02 | 290.99 | timeout |
| order_06 | kitchen → bedroom → living_room | 0° | PASS | 0 | 0 | 109.96 | 178.61 |  |
| heading_01 | living_room → bedroom → kitchen | -30° | FAIL | 0 | 0 | 20.02 | 25.64 | saved-map localization or occupancy grid unavailable |
| heading_02 | living_room → bedroom → kitchen | 30° | FAIL | 0 | 0 | 20.02 | 25.64 | saved-map localization or occupancy grid unavailable |
| heading_03 | living_room → bedroom → kitchen | 60° | FAIL | 0 | 0 | 20.02 | 25.59 | saved-map localization or occupancy grid unavailable |
| heading_04 | living_room → bedroom → kitchen | 180° | FAIL | 0 | 0 | 20.02 | 25.64 | saved-map localization or occupancy grid unavailable |

仿真耗时（包含失败）：均值 86.92 s，中位数 111.16 s，P95 150.02 s，最大 150.02 s。

墙钟耗时（包含失败）：均值 163.97 s，中位数 203.61 s，P95 301.75 s，最大 301.75 s。
