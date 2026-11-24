"""避障几何、解析逆解与控制器行为测试。"""

import unittest

import numpy as np


class AvoidanceGeometryTests(unittest.TestCase):
    def test_point_segment_distance_returns_perpendicular_distance(self):
        """垂足落在线段内部时，应返回点到直线的垂直距离。"""
        from mujoco_arm_lab.avoidance import point_segment_distance

        distance = point_segment_distance(
            np.array([0.5, 0.2, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(distance, 0.2)

    def test_point_segment_distance_clamps_to_segment_endpoint(self):
        """垂足超出线段时，应夹到最近端点而不是无限延长线。"""
        from mujoco_arm_lab.avoidance import point_segment_distance

        distance = point_segment_distance(
            np.array([2.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(distance, 1.0)

    def test_choose_waypoint_returns_none_for_clear_path(self):
        """直线路径满足安全间距时，不应额外插入绕行点。"""
        from mujoco_arm_lab.avoidance import choose_waypoint

        waypoint = choose_waypoint(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            obstacle=np.array([0.5, 0.3, 0.0]),
            clearance=0.2,
        )

        self.assertIsNone(waypoint)

    def test_choose_waypoint_moves_path_clear_of_blocked_obstacle(self):
        """障碍物挡住直线时，两段折线都应满足清距要求。"""
        from mujoco_arm_lab.avoidance import choose_waypoint, point_segment_distance

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 0.0, 0.0])
        obstacle = np.array([0.5, 0.0, 0.0])
        clearance = 0.2

        waypoint = choose_waypoint(start, end, obstacle, clearance)

        self.assertIsNotNone(waypoint)
        self.assertGreaterEqual(point_segment_distance(obstacle, start, waypoint), clearance)
        self.assertGreaterEqual(point_segment_distance(obstacle, waypoint, end), clearance)


class ElbowUpInverseKinematicsTests(unittest.TestCase):
    def test_elbow_up_solution_reaches_target_and_lifts_first_link(self):
        """肘部上抬解应到达目标，并让第一根连杆与障碍物保持距离。"""
        import mujoco

        from mujoco_arm_lab.avoidance import AvoidanceReacher, solve_elbow_up
        from mujoco_arm_lab.control import sample_target

        rng = np.random.default_rng(0)
        targets = [sample_target(rng) for _ in range(10)]
        target = targets[9]
        reacher = AvoidanceReacher()
        reacher.reset(target)

        joints = solve_elbow_up(target)
        reacher.data.qpos[reacher.q_idx] = joints
        mujoco.mj_forward(reacher.model, reacher.data)

        self.assertLess(reacher.error_to(target), 0.01)
        self.assertGreater(reacher.minimum_arm_clearance(), 0.18)


class AvoidanceControllerTests(unittest.TestCase):
    def test_run_avoidance_reaches_known_collision_target_without_contact(self):
        """已知会碰撞的目标在避障流程中应成功且不发生接触。"""
        from mujoco_arm_lab.control import sample_target
        from mujoco_arm_lab.avoidance import AvoidanceReacher

        rng = np.random.default_rng(0)
        targets = [sample_target(rng) for _ in range(2)]
        reacher = AvoidanceReacher()

        error, _, collision, _ = reacher.run_avoidance(targets[1])

        self.assertFalse(collision)
        self.assertLess(error, 0.005)

    def test_negative_y_target_takes_direct_route(self):
        """左前方目标不应触发当前只覆盖右前方低位的绕行分支。"""
        from mujoco_arm_lab.control import sample_target
        from mujoco_arm_lab.avoidance import AvoidanceReacher

        rng = np.random.default_rng(0)
        target = sample_target(rng)
        reacher = AvoidanceReacher()
        reacher.reset(target)

        self.assertEqual(reacher.plan_joint_waypoints(target), [])

    def test_run_avoidance_reaches_low_positive_y_target_without_contact(self):
        """右前方低位目标应经避障路径到达且保持无碰撞。"""
        from mujoco_arm_lab.control import sample_target
        from mujoco_arm_lab.avoidance import AvoidanceReacher

        rng = np.random.default_rng(0)
        targets = [sample_target(rng) for _ in range(10)]
        reacher = AvoidanceReacher()

        error, _, collision, _ = reacher.run_avoidance(targets[9])

        self.assertFalse(collision)
        self.assertLess(error, 0.005)

    def test_evaluate_avoidance_returns_comparable_metrics(self):
        """避障评测应返回与基线完全一致的汇总字段。"""
        from mujoco_arm_lab.avoidance import evaluate_avoidance

        summary, episodes = evaluate_avoidance(n=1, seed=0)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(
            set(summary),
            {
                "n",
                "success_count",
                "success_rate",
                "collision_count",
                "collision_rate",
                "mean_error",
                "mean_duration",
                "timeout_count",
            },
        )

    def test_evaluate_avoidance_applies_custom_obstacle_configuration(self):
        """自定义障碍物位置和半径应写入每条回合记录。"""
        from mujoco_arm_lab.avoidance import evaluate_avoidance

        position = np.array([0.40, -0.15, 0.10])
        _, episodes = evaluate_avoidance(
            n=1,
            seed=0,
            obstacle_position=position,
            obstacle_radius=0.05,
        )

        np.testing.assert_allclose(episodes[0]["obstacle_position"], position)
        self.assertAlmostEqual(episodes[0]["obstacle_radius"], 0.05)


if __name__ == "__main__":
    unittest.main()
