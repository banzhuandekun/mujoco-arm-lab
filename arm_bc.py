"""
行为克隆（BC）策略 —— 用一个神经网络模仿 IK 专家的动作。

为什么不是纯强化学习：
  这个项目里 PPO 和 SAC 都试过了，在这台 CPU 机器上样本效率远远不够
  （PPO 8 万步成功率仍是 0，SAC 只有约 55 步/秒，训到有效果要一小时以上）。
  于是改成从 IK 专家蒸馏：目标同样是「学出一个策略」，
  但换成监督学习，几分钟就能收敛，而且效果接近专家本身。

最关键的一条经验（踩坑换来的）：
  专家数据必须包含「偏离状态」。
  只用 IK 的干净轨迹训练，成功率只有 62%；在轨迹上注入随机扰动、
  让专家演示如何从偏离状态纠正回来，成功率升到 98%。
  注意此时训练 MSE 反而更高（1.07e-2 vs 5.4e-3），但实际成功率大幅提升 ——
  决定泛化能力的是数据覆盖范围，不是拟合精度。
"""

import numpy as np
import torch
import torch.nn as nn


class BCPolicyNet(nn.Module):
    """10 维观测 → 3 维动作的小 MLP"""

    def __init__(self, obs_dim=10, act_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),   # Tanh 把输出限制在 [-1,1]，和动作空间一致
        )

    def forward(self, x):
        return self.net(x)


class BCPolicy:
    """包装成和 stable-baselines3 一样的 predict() 接口，方便和 PPO 用同一套评测代码"""

    def __init__(self, net):
        self.net = net

    @classmethod
    def load(cls, path):
        net = BCPolicyNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        return cls(net)

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            action = self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        return action.numpy(), None
