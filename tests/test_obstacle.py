"""障碍物模型、接触检测与基线评测测试。"""

import unittest

import mujoco
import numpy as np


class ObstacleScenarioTests(unittest.TestCase):
    def test_obstacle_model_contains_one_collidable_obstacle(self):
        """障碍物 MJCF 中应存在可参与双向接触的 geom。"""
        from mujoco_arm_lab.obstacle import OBSTACLE_GEOM_NAME, ObstacleReacher

        reacher = ObstacleReacher()
        geom_id = mujoco.mj_name2id(
            reacher.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            OBSTACLE_GEOM_NAME,
        )

        self.assertGreaterEqual(geom_id, 0)
        self.assertGreater(reacher.model.geom_contype[geom_id], 0)
        self.assertGreater(reacher.model.geom_conaffinity[geom_id], 0)

    def test_collision_detection_ignores_distant_obstacle(self):
        """障碍物远离机械臂时不应误报接触。"""
        from mujoco_arm_lab.obstacle import ObstacleReacher

        reacher = ObstacleReacher()
        reacher.reset(np.array([0.35, 0.0, 0.30]))
        reacher.set_obstacle(np.array([-0.8, 0.8, 0.8]), radius=0.05)
        mujoco.mj_forward(reacher.model, reacher.data)

        self.assertFalse(reacher.has_collision())

    def test_collision_detection_finds_obstacle_touching_end_effector(self):
        """障碍物与末端重合时应检测到接触。"""
        from mujoco_arm_lab.obstacle import ObstacleReacher

        reacher = ObstacleReacher()
        reacher.reset(np.array([0.35, 0.0, 0.30]))
        reacher.set_obstacle(reacher.ee_pos, radius=0.05)
        mujoco.mj_forward(reacher.model, reacher.data)

        self.assertTrue(reacher.has_collision())

    def test_reset_preserves_configured_obstacle(self):
        """基础控制器重置机械臂时不能把障碍物配置改回默认值。"""
        from mujoco_arm_lab.obstacle import ObstacleReacher

        reacher = ObstacleReacher()
        position = np.array([0.30, 0.10, 0.18])
        reacher.set_obstacle(position, radius=0.09)

        reacher.reset(np.array([0.35, 0.0, 0.30]))

        np.testing.assert_allclose(
            reacher.data.mocap_pos[reacher.obstacle_mocap_id],
            position,
        )
        self.assertAlmostEqual(
            reacher.model.geom_size[reacher.obstacle_geom_id, 0],
            0.09,
        )

    def test_default_scenario_contains_detectable_collision(self):
        """默认场景应保留一个可稳定复现碰撞的目标点。"""
        from mujoco_arm_lab.control import sample_target
        from mujoco_arm_lab.obstacle import ObstacleReacher

        rng = np.random.default_rng(0)
        targets = [sample_target(rng) for _ in range(2)]
        reacher = ObstacleReacher()

        error, _, collision, first_contact_step = reacher.run_obstacle(targets[1])

        self.assertTrue(collision)
        self.assertIsNotNone(first_contact_step)
        self.assertLess(error, 0.005)

    def test_evaluate_returns_required_metrics_for_each_episode(self):
        """基线评测的汇总字段和单轮字段应保持公开接口稳定。"""
        from mujoco_arm_lab.obstacle import evaluate_ik

        summary, episodes = evaluate_ik(n=1, seed=0)

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
        self.assertEqual(
            set(episodes[0]),
            {
                "index",
                "target",
                "error",
                "steps",
                "duration",
                "success",
                "collision",
                "timeout",
            },
        )


if __name__ == "__main__":
    unittest.main()
