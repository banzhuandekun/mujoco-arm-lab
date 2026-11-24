# 脚本索引

所有命令都在仓库根目录执行。首次使用先完成：

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

## 学习与演示

| 脚本 | 用途 | 输出 |
|---|---|---|
| `scripts/01_first_motion.py` | 用正弦轨迹验证模型和渲染链路 | `outputs/01_first_motion.*` |
| `scripts/02_reach_target.py` | 随机目标 DLS-IK 评测，可录视频 | 终端指标、`outputs/02_reach_target.*` |
| `scripts/03_interactive_viewer.py` | 连续作业的交互式 3D 查看器 | 实时窗口 |
| `scripts/05_compare.py` | 在同一批目标上比较 IK 与 BC | 终端对比表 |
| `scripts/06_train_bc.py` | 采集带扰动的专家数据并训练 BC | `outputs/bc_policy.pt` |

常用命令：

```powershell
.\.venv\Scripts\python.exe scripts\01_first_motion.py
.\.venv\Scripts\python.exe scripts\02_reach_target.py --n-eval 100 --no-video
.\.venv\Scripts\python.exe scripts\03_interactive_viewer.py
.\.venv\Scripts\python.exe scripts\06_train_bc.py
.\.venv\Scripts\python.exe scripts\05_compare.py --n-eval 100
```

`03` 会打开图形窗口；`06` 默认用 300 条专家轨迹、训练 180 轮，完整运行约数分钟
到十几分钟，具体取决于机器。快速试跑：

```powershell
.\.venv\Scripts\python.exe scripts\06_train_bc.py --episodes 150 --epochs 90
```

## 实验

| 脚本 | 用途 | 默认规模 |
|---|---|---|
| `experiments/rl/04_train_ppo.py` | 保留的 PPO 样本效率实验，当前未收敛 | 200,000 步 |
| `experiments/obstacle/07_obstacle_eval.py` | 障碍场景原始 DLS-IK 基线 | 50 个目标 |
| `experiments/obstacle/08_avoidance_eval.py` | 避障前后同目标点对比 | 50 个目标 |
| `experiments/obstacle/09_obstacle_matrix.py` | 障碍物位置 × 半径矩阵 | 每格 100 个目标 |

```powershell
.\.venv\Scripts\python.exe experiments\rl\04_train_ppo.py --timesteps 50000
.\.venv\Scripts\python.exe experiments\obstacle\07_obstacle_eval.py --n-eval 50 --seed 0
.\.venv\Scripts\python.exe experiments\obstacle\08_avoidance_eval.py --n-eval 50 --seed 0
.\.venv\Scripts\python.exe experiments\obstacle\09_obstacle_matrix.py --n-eval 100 --seed 0
```

## 工具与测试

```powershell
# 把已有 MP4 转成 README 用 GIF
.\.venv\Scripts\python.exe tools\make_gif.py

# 全量测试
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tools/make_gif.py` 只处理 `outputs` 中已经存在的视频。缺失的视频会跳过并给出提示，
不会自动启动渲染。
