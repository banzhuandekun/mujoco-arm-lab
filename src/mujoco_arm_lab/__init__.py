"""MuJoCo 三自由度机械臂实验包。

包内只放可复用的控制、环境、评测与策略代码；面向用户的入口脚本分别放在
``scripts/``、``experiments/`` 和 ``tools/``。本模块导出最常用的公共对象，
便于交互式调试：``from mujoco_arm_lab import Reacher``。
"""

from .bc import BCPolicy, BCPolicyNet
from .control import Reacher, load_model, sample_target
from .env import ArmReachEnv

__all__ = [
    "ArmReachEnv",
    "BCPolicy",
    "BCPolicyNet",
    "Reacher",
    "load_model",
    "sample_target",
]
