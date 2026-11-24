# 评测口径

## 为什么需要统一口径

同一个控制器在“每轮重置姿态”和“连续作业不重置”两种工况下的成功率差别很大。
不写清场景、目标点、时间预算和判定阈值，成功率数字不能互相比较。

本仓库的默认口径：

| 项目 | 定义 |
|---|---|
| 目标点 | `sample_target()` 在可达球壳内采样 |
| 到达阈值 | 末端误差 `< 0.005 m` |
| 单轮预算 | 最多 `4000` 个仿真步 |
| 仿真步长 | 读取 MJCF 的 `model.opt.timestep`，默认 `0.002 s` |
| 单轮耗时 | `steps × timestep` |
| 成功率 | `success_count / n` |
| 碰撞率 | 发生过至少一次障碍物接触的回合数 `/ n` |
| 超时率 | 使用满 `4000` 步的回合数；当前表格只报告次数 |

成功率回答“最终有没有到达”，碰撞率回答“过程中安不安全”。两者必须同时报告。
一个回合可能碰撞但仍然成功，也可能安全但超时失败。

## 单轮记录

`mujoco_arm_lab.evaluation.make_episode()` 统一生成以下字段：

| 字段 | 含义 |
|---|---|
| `index` | 当前回合从 0 开始的序号 |
| `target` | 目标点世界坐标 |
| `error` | 回合结束时的末端距离，单位米 |
| `steps` | 实际仿真步数 |
| `duration` | 实际仿真时间，单位秒 |
| `success` | 是否进入到达阈值 |
| `collision` | 是否与障碍物发生过接触 |
| `timeout` | 是否耗尽单轮步数预算 |

避障记录额外保留 `waypoint_step`、`obstacle_position` 和 `obstacle_radius`，用于定位
具体失败配置。

## 结果解释

README 中的主要结果分为三类：

1. **无障碍 DLS-IK**：验证基础控制器和渲染链路。
2. **有障碍未避障**：测清原始控制器的碰撞方式和失败范围。
3. **避障前后对比**：保持同一随机种子和目标数量，比较成功率、碰撞率和耗时。

“右前方低位单球成功率 100%”不等于通用避障能力。位置矩阵显示，当前策略换到
左前方或上方后，碰撞率可能完全不变，甚至会让成功率下降。

## 复现命令

```powershell
# 无障碍，逐轮重置姿态
.\.venv\Scripts\python.exe scripts\02_reach_target.py --n-eval 50 --seed 0 --no-video

# 有障碍，原始 DLS-IK
.\.venv\Scripts\python.exe experiments\obstacle\07_obstacle_eval.py --n-eval 50 --seed 0

# 同一批目标上的避障前后对比
.\.venv\Scripts\python.exe experiments\obstacle\08_avoidance_eval.py --n-eval 50 --seed 0

# 5 个位置 × 2 个半径
.\.venv\Scripts\python.exe experiments\obstacle\09_obstacle_matrix.py --n-eval 100 --seed 0
```

修改控制参数后，旧结果不能直接沿用。至少重新运行对应评测，并在提交说明中写明
目标数量、随机种子和运行命令。
