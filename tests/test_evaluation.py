"""公共评测汇总逻辑的单元测试。"""

import unittest

from mujoco_arm_lab.evaluation import make_episode, summarize_episodes


class EvaluationSummaryTests(unittest.TestCase):
    """验证成功、碰撞、超时和耗时使用统一口径。"""

    def test_make_episode_computes_duration_and_preserves_extra_fields(self):
        """单轮记录应按时间步换算耗时，并保留策略专属字段。"""
        episode = make_episode(
            index=0,
            target=[0.4, 0.1, 0.2],
            error=0.004,
            steps=250,
            timestep=0.002,
            success=True,
            collision=False,
            timeout=False,
            waypoint_step=120,
        )

        self.assertAlmostEqual(episode["duration"], 0.5)
        self.assertEqual(episode["waypoint_step"], 120)

    def test_collision_is_independent_from_success(self):
        """碰撞但到达的目标仍计成功，同时计一次碰撞。"""
        episodes = [
            make_episode(
                index=0,
                target=[0.4, 0.0, 0.2],
                error=0.004,
                steps=100,
                timestep=0.002,
                success=True,
                collision=True,
                timeout=False,
            ),
            make_episode(
                index=1,
                target=[0.3, 0.0, 0.2],
                error=0.02,
                steps=4000,
                timestep=0.002,
                success=False,
                collision=False,
                timeout=True,
            ),
        ]

        summary = summarize_episodes(episodes)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["collision_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertAlmostEqual(summary["mean_duration"], 4.1)

    def test_empty_episode_list_is_rejected(self):
        """空输入没有可定义的均值，应明确拒绝。"""
        with self.assertRaises(ValueError):
            summarize_episodes([])


if __name__ == "__main__":
    unittest.main()
