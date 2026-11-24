"""评测记录与指标汇总的公共实现。

``obstacle``、``avoidance`` 和后续控制实验共享同一套指标定义。集中维护可以
避免某个实验改了“超时”或“成功率”的口径，却忘记同步其他表格。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def make_episode(
    *,
    index: int,
    target: np.ndarray,
    error: float,
    steps: int,
    timestep: float,
    success: bool,
    collision: bool,
    timeout: bool,
    **extra: Any,
) -> dict[str, Any]:
    """创建一条统一格式的单目标评测记录。

    参数使用关键字传入，避免把误差、步数和耗时等相近的数值型参数写错位置。
    额外字段会原样并入记录，例如避障策略的 ``waypoint_step``。
    """
    episode = {
        "index": index,
        "target": np.asarray(target, dtype=float),
        "error": float(error),
        "steps": int(steps),
        "duration": float(steps * timestep),
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
    }
    episode.update(extra)
    return episode


def summarize_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    """汇总单目标记录，返回成功率、碰撞率、误差与耗时指标。

    成功率按 ``success`` 字段计算；超时和碰撞保持独立计数。这样“碰到障碍物但
    最终到达”不会被误判成失败，也不会被成功率掩盖掉安全性问题。
    """
    if not episodes:
        raise ValueError("episodes 不能为空")

    errors = np.array([episode["error"] for episode in episodes], dtype=float)
    durations = np.array([episode["duration"] for episode in episodes], dtype=float)
    successes = np.array([episode["success"] for episode in episodes], dtype=bool)
    collisions = np.array([episode["collision"] for episode in episodes], dtype=bool)
    timeouts = np.array([episode["timeout"] for episode in episodes], dtype=bool)
    n = len(episodes)

    return {
        "n": n,
        "success_count": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "collision_count": int(collisions.sum()),
        "collision_rate": float(collisions.mean()),
        "mean_error": float(errors.mean()),
        "mean_duration": float(durations.mean()),
        "timeout_count": int(timeouts.sum()),
    }
