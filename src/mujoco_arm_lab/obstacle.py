"""带静态球形障碍物的 IK 基线评测。

这一阶段只负责把问题测准：记录到达成功率、碰撞率和耗时，
不包含避障控制。后续避障算法必须和这里使用相同的目标点、判据与时间预算。
"""

import mujoco
import numpy as np

from .control import IK_EVERY, MAX_STEPS, TOL, Reacher, sample_target
from .evaluation import make_episode, summarize_episodes
from .paths import OBSTACLE_MODEL_PATH

OBSTACLE_GEOM_NAME = "obstacle_geom"
DEFAULT_OBSTACLE_POSITION = np.array([0.40, 0.15, 0.10])
DEFAULT_OBSTACLE_RADIUS = 0.07


class ObstacleReacher(Reacher):
    """在现有 DLS-IK 控制器上增加障碍物接触检测。"""

    def __init__(self):
        """加载障碍物场景，并缓存接触检测需要的 geom 与 mocap 索引。"""
        super().__init__(OBSTACLE_MODEL_PATH)
        self.obstacle_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            OBSTACLE_GEOM_NAME,
        )
        obstacle_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "obstacle",
        )
        self.obstacle_mocap_id = self.model.body_mocapid[obstacle_body_id]
        self.contact_steps = 0
        self.first_contact_step = None
        self.set_obstacle(DEFAULT_OBSTACLE_POSITION, DEFAULT_OBSTACLE_RADIUS)

    def set_obstacle(self, position, radius=DEFAULT_OBSTACLE_RADIUS):
        """移动障碍物并修改半径，调用后立即刷新碰撞状态。"""
        self.obstacle_position = np.asarray(position, dtype=float)
        self.obstacle_radius = float(radius)
        self.data.mocap_pos[self.obstacle_mocap_id] = self.obstacle_position
        self.model.geom_size[self.obstacle_geom_id, 0] = self.obstacle_radius
        mujoco.mj_forward(self.model, self.data)

    def has_collision(self):
        """当前仿真时刻，机械臂是否与障碍物接触。"""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.obstacle_geom_id in (contact.geom1, contact.geom2):
                return True
        return False

    def reset(self, target):
        """重置机械臂，同时恢复当前配置的障碍物位置与半径。"""
        super().reset(target)
        self.set_obstacle(self.obstacle_position, self.obstacle_radius)
        self.contact_steps = 0
        self.first_contact_step = None
        return self.q_target.copy()

    def run_obstacle(self, target, max_steps=MAX_STEPS):
        """运行一次带碰撞记录的到达尝试。"""
        self.reset(target)
        steps = 0
        collided = False

        for step in range(max_steps):
            if step % IK_EVERY == 0:
                if self.error_to(target) < TOL:
                    break
                self.ik_step(target)

            mujoco.mj_step(self.model, self.data)
            steps += 1

            if self.has_collision():
                collided = True
                self.contact_steps += 1
                if self.first_contact_step is None:
                    self.first_contact_step = steps

        return self.error_to(target), steps, collided, self.first_contact_step


def evaluate_ik(n, seed=0, obstacle_position=None, obstacle_radius=None):
    """用固定种子评测无障碍目标点，并统计障碍场景下的碰撞。"""
    rng = np.random.default_rng(seed)
    reacher = ObstacleReacher()
    dt = reacher.model.opt.timestep
    position = DEFAULT_OBSTACLE_POSITION if obstacle_position is None else obstacle_position
    radius = DEFAULT_OBSTACLE_RADIUS if obstacle_radius is None else obstacle_radius
    episodes = []

    for index in range(n):
        target = sample_target(rng)
        reacher.set_obstacle(position, radius)
        error, steps, collision, _ = reacher.run_obstacle(target)
        episodes.append(
            make_episode(
                index=index,
                target=target,
                error=error,
                steps=steps,
                timestep=dt,
                success=error < TOL,
                collision=collision,
                timeout=steps >= MAX_STEPS,
            )
        )

    return summarize_episodes(episodes), episodes
