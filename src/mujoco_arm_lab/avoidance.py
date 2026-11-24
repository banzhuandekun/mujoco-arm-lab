"""基于安全关节构型的静态单球避障。"""

import mujoco
import numpy as np

from .control import IK_EVERY, MAX_DQ, MAX_STEPS, TOL, sample_target
from .evaluation import make_episode, summarize_episodes
from .obstacle import (
    DEFAULT_OBSTACLE_POSITION,
    DEFAULT_OBSTACLE_RADIUS,
    ObstacleReacher,
)

ARM_CLEARANCE = 0.18
JOINT_TOL = 0.04
SAFE_LIFT_JOINTS = np.array([0.0, -1.0, -0.2])
LINK1_LENGTH = 0.4
LINK2_LENGTH = 0.3
SHOULDER_HEIGHT = 0.1


def point_segment_distance(point, start, end):
    """点到有限线段的最短距离。"""
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))

    projection = float((point - start) @ segment / length_squared)
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def choose_waypoint(start, end, obstacle, clearance):
    """直线路径穿过膨胀障碍物时，在障碍物正上方生成安全中间点。"""
    if point_segment_distance(obstacle, start, end) >= clearance:
        return None
    waypoint = np.asarray(obstacle, dtype=float).copy()
    waypoint[2] += clearance * 1.25
    return waypoint


def solve_elbow_up(target):
    """求解三自由度机械臂的肘部上抬解析逆解。"""
    target = np.asarray(target, dtype=float)
    radial = float(np.linalg.norm(target[:2]))
    height = float(target[2] - SHOULDER_HEIGHT)
    cosine = (
        radial ** 2 + height ** 2 - LINK1_LENGTH ** 2 - LINK2_LENGTH ** 2
    ) / (2.0 * LINK1_LENGTH * LINK2_LENGTH)
    cosine = float(np.clip(cosine, -1.0, 1.0))

    elbow_angle = float(np.arccos(cosine))
    shoulder = -(
        float(np.arctan2(height, radial))
        - float(
            np.arctan2(
                -LINK2_LENGTH * np.sin(elbow_angle),
                LINK1_LENGTH + LINK2_LENGTH * np.cos(elbow_angle),
            )
        )
    )
    yaw = float(np.arctan2(target[1], target[0]))
    return np.array([yaw, shoulder, elbow_angle])


class AvoidanceReacher(ObstacleReacher):
    """机械臂离障碍物过近时，先抬臂转向，再用肘部上抬构型追目标。"""

    def __init__(self, arm_clearance=ARM_CLEARANCE):
        """建立避障控制器，并缓存两根连杆在 MuJoCo 中的 body 索引。"""
        super().__init__()
        self.arm_clearance = float(arm_clearance)
        self.link1_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "link1",
        )
        self.link2_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "link2",
        )

    def minimum_arm_clearance(self):
        """两条机械臂连杆到障碍物球心的最小距离。"""
        link1 = self.data.xpos[self.link1_body_id]
        link2 = self.data.xpos[self.link2_body_id]
        return min(
            point_segment_distance(self.obstacle_position, link1, link2),
            point_segment_distance(self.obstacle_position, link2, self.ee_pos),
        )

    def plan_joint_waypoints(self, target):
        """需要避障时，先生成抬臂姿态，再生成转向目标的姿态。"""
        if target[1] <= 0.0:
            return []

        blocked = choose_waypoint(
            self.ee_pos,
            target,
            self.obstacle_position,
            self.arm_clearance,
        )
        if blocked is None and self.minimum_arm_clearance() >= self.arm_clearance:
            return []

        current = self.data.qpos[self.q_idx]
        target_yaw = float(np.arctan2(target[1], target[0]))
        target_yaw = float(np.clip(target_yaw, self.q_lo[0], self.q_hi[0]))
        lift = SAFE_LIFT_JOINTS.copy()
        lift[0] = current[0]
        turn = SAFE_LIFT_JOINTS.copy()
        turn[0] = target_yaw
        return [lift, turn]

    def _advance_joint_target(self, goal, budget):
        """平滑移动关节目标，返回本阶段消耗步数及是否发生碰撞。"""
        collided = False
        steps = 0
        for _ in range(budget):
            if (
                np.max(np.abs(self.data.qpos[self.q_idx] - goal)) < JOINT_TOL
                and np.max(np.abs(self.q_target - goal)) < JOINT_TOL
            ):
                break

            if steps % IK_EVERY == 0:
                delta = np.clip(goal - self.q_target, -MAX_DQ, MAX_DQ)
                self.q_target = np.clip(self.q_target + delta, self.q_lo, self.q_hi)
                self.data.ctrl[:] = self.q_target

            mujoco.mj_step(self.model, self.data)
            steps += 1
            if self.has_collision():
                collided = True
                self.contact_steps += 1
                if self.first_contact_step is None:
                    self.first_contact_step = steps
        return steps, collided

    def run_avoidance(self, target, max_steps=MAX_STEPS):
        """依次经过安全关节姿态，再恢复原来的 DLS-IK 追目标。"""
        self.reset(target)
        waypoints = self.plan_joint_waypoints(target)
        steps = 0
        collided = False
        waypoint_step = None

        for index, waypoint in enumerate(waypoints):
            remaining = max_steps - steps
            if remaining <= 0:
                break
            stage_steps, stage_collided = self._advance_joint_target(waypoint, remaining)
            steps += stage_steps
            collided = collided or stage_collided
            if index == len(waypoints) - 1:
                waypoint_step = steps

        if waypoints:
            target_joints = np.clip(
                solve_elbow_up(target),
                self.q_lo,
                self.q_hi,
            )
            remaining = max_steps - steps
            stage_steps, stage_collided = self._advance_joint_target(
                target_joints,
                remaining,
            )
            steps += stage_steps
            collided = collided or stage_collided
            waypoint_step = steps

        while steps < max_steps:
            if steps % IK_EVERY == 0:
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

        return self.error_to(target), steps, collided, waypoint_step


def evaluate_avoidance(
    n,
    seed=0,
    obstacle_position=None,
    obstacle_radius=None,
):
    """用固定种子评测关节空间避障策略。"""
    rng = np.random.default_rng(seed)
    reacher = AvoidanceReacher()
    dt = reacher.model.opt.timestep
    position = (
        DEFAULT_OBSTACLE_POSITION
        if obstacle_position is None
        else np.asarray(obstacle_position, dtype=float)
    )
    radius = (
        DEFAULT_OBSTACLE_RADIUS
        if obstacle_radius is None
        else float(obstacle_radius)
    )
    reacher.set_obstacle(position, radius)
    episodes = []

    for index in range(n):
        target = sample_target(rng)
        error, steps, collision, waypoint_step = reacher.run_avoidance(target)
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
                waypoint_step=waypoint_step,
                obstacle_position=position.copy(),
                obstacle_radius=radius,
            )
        )

    return summarize_episodes(episodes), episodes
