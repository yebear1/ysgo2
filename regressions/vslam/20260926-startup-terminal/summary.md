# VSLAM 批量回归

状态：完成；已完成 10/10。
成功：10/10（所有预定样本作分母，未完成不算成功）。
已知接触事件：0；卡住事件：0。
缺少接触/卡住指标的运行：0/0。

卡住定义：导航激活且视觉有效时，至少 10 仿真秒位移不足 0.15 m、航向变化不足 20°；遮挡/失定位时间单独排除。
有意转身不计为卡住；长时间低效绕行仍可能超时，但不一定符合此静止判据。

| 运行 | 顺序（最后均回起点） | 出发朝向 | 结果 | 碰撞 | 卡住 | 仿真秒 | 墙钟秒 | 原因 |
|---|---|---:|---|---:|---:|---:|---:|---|
| heading_01 | living_room → bedroom → kitchen | -30° | PASS | 0 | 0 | 136.42 | 262.45 |  |
| heading_02 | living_room → bedroom → kitchen | 30° | PASS | 0 | 0 | 134.46 | 259.69 |  |
| heading_03 | living_room → bedroom → kitchen | 60° | PASS | 0 | 0 | 121.58 | 236.51 |  |
| heading_04 | living_room → bedroom → kitchen | 180° | PASS | 0 | 0 | 130.70 | 251.72 |  |
| order_01 | living_room → bedroom → kitchen | 0° | PASS | 0 | 0 | 117.74 | 228.97 |  |
| order_02 | living_room → kitchen → bedroom | 0° | PASS | 0 | 0 | 129.64 | 249.80 |  |
| order_03 | bedroom → living_room → kitchen | 0° | PASS | 0 | 0 | 148.26 | 287.36 |  |
| order_04 | bedroom → kitchen → living_room | 0° | PASS | 0 | 0 | 105.28 | 205.77 |  |
| order_05 | kitchen → living_room → bedroom | 0° | PASS | 0 | 0 | 145.96 | 284.77 |  |
| order_06 | kitchen → bedroom → living_room | 0° | PASS | 0 | 0 | 111.94 | 218.96 |  |

仿真耗时（包含失败）：均值 128.20 s，中位数 130.17 s，P95 148.26 s，最大 148.26 s。

墙钟耗时（包含失败）：均值 248.60 s，中位数 250.76 s，P95 287.36 s，最大 287.36 s。
