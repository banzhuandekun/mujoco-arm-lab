"""
Gymnasium 环境 —— 把机械臂场景包装成强化学习环境。

包装成标准接口后，stable-baselines3 这类 RL 库就能直接在上面训练。

动作空间：3 维连续，表示三个关节目标角的增量，范围 [-1, 1]
          实际增量 = action × MAX_ACTION_DQ
观测空间：10 维（已缩放到大致 [-1, 1]）
  [0:3]  三个关节角
  [3:6]  三个关节角速度
  [6:9]  末端指向目标的向量（目标 − 末端），世界坐标
  [9]    末端到目标的距离
奖励：
  每一步         (上一步距离 − 当前距离) × 20 − 0.01
  到达目标       额外 +20 并结束回合
  超过时间上限    回合截断（拿不到那 +20）

奖励为什么这么设计（踩过坑）：
  最初写的是「每一步 −距离」，结果训练 5 万步成功率还是 0。
  原因是跑满一回合累计奖励约 −616，而到达奖励只有 +10，
  「够到了」和「没够到」在回报里几乎分不出来，PPO 学不动。
  改成「距离的进步量」后，每一步靠近都有明确的正反馈，
  超时（原地打转）则持续吃时间惩罚，两种行为才拉得开差距。
  系数别乱配：每决策步平均前进约 0.005 m，若 PROGRESS_SCALE 取 10、
  TIME_PENALTY 取 0.05，两者正好抵消，「在前进」和「原地不动」就没区别了。

控制频率：每个 RL 决策对应 ACTION_REPEAT 个仿真步（默认 5 步 = 100 Hz），
和 02 里 IK 的控制频率保持一致 —— 频率不一样的话对比就没意义了。
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from arm_control import IK_EVERY, TOL, Reacher, sample_target

ACTION_REPEAT = IK_EVERY      # 每个决策对应的仿真步数
MAX_ACTION_DQ = 0.02          # action = 1.0 时每个关节的最大增量（rad）
MAX_DECISION_STEPS = 800      # 单回合最多 800 次决策 = 4000 仿真步 = 8 s
PROGRESS_SCALE = 20.0         # 距离进步量在奖励里的权重
TIME_PENALTY = 0.01           # 每步固定扣一点，逼它尽快到达
REACH_BONUS = 20.0            # 到达目标的一次性奖励

# 观测缩放：各分量除以这个尺度，让数值都落在 [-1, 1] 附近，PPO 训练更稳
OBS_SCALE = np.array(
    [3.0, 3.0, 3.0,       # 关节角
     5.0, 5.0, 5.0,       # 关节角速度
     0.7, 0.7, 0.7,       # 末端到目标的向量
     0.7]                 # 距离
)


class ArmReachEnv(gym.Env):
    """三自由度机械臂到达随机目标点"""

    metadata = {"render_modes": []}

    def __init__(self, seed=None, progress_scale=None, time_penalty=None,
                 reach_bonus=None):
        super().__init__()
        self.reacher = Reacher()
        self.model = self.reacher.model
        self.data = self.reacher.data
        self.rng = np.random.default_rng(seed)

        # 奖励系数做成实例属性，方便调参实验时逐个覆盖
        self.progress_scale = PROGRESS_SCALE if progress_scale is None else float(progress_scale)
        self.time_penalty = TIME_PENALTY if time_penalty is None else float(time_penalty)
        self.reach_bonus = REACH_BONUS if reach_bonus is None else float(reach_bonus)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(10,), dtype=np.float32)

        self.target = None
        self.decision_steps = 0

    # ---------- 内部工具 ----------

    def _raw_obs(self):
        q = self.data.qpos[self.reacher.q_idx]
        qd = self.data.qvel[self.reacher.d_idx]
        delta = self.target - self.reacher.ee_pos
        return np.concatenate([q, qd, delta, [np.linalg.norm(delta)]])

    def _obs(self):
        return (self._raw_obs() / OBS_SCALE).astype(np.float32)

    # ---------- Gymnasium 接口 ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "target" in options:
            # 评测时传入固定目标点，保证 IK 和 PPO 面对的是同一批点
            self.target = np.asarray(options["target"], dtype=np.float64)
        else:
            self.target = sample_target(self.rng)

        self.reacher.reset(self.target)   # 回到初始姿态，并把目标点摆到 target
        self.decision_steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        prev_dist = self.reacher.error_to(self.target)

        # 动作 = 关节目标角的增量，累积到当前指令上（和 IK 一样是「累积式」控制）
        self.reacher.q_target = np.clip(
            self.reacher.q_target + action * MAX_ACTION_DQ,
            self.reacher.q_lo, self.reacher.q_hi)
        self.data.ctrl[:] = self.reacher.q_target

        for _ in range(ACTION_REPEAT):
            mujoco.mj_step(self.model, self.data)

        dist = self.reacher.error_to(self.target)
        reward = (prev_dist - dist) * self.progress_scale - self.time_penalty
        terminated = False
        if dist < TOL:
            reward += self.reach_bonus
            terminated = True

        self.decision_steps += 1
        truncated = self.decision_steps >= MAX_DECISION_STEPS

        info = {"distance": dist, "is_success": terminated}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        raise NotImplementedError("这个环境不做渲染，要看画面请用 02 或 03")
