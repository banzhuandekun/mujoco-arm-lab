"""障碍物位置与半径矩阵的编排测试。"""

import unittest


class ObstacleMatrixTests(unittest.TestCase):
    def test_matrix_compares_baseline_and_avoidance_for_each_case(self):
        """每个位置和半径组合都应同时生成基线与避障结果。"""
        from mujoco_arm_lab.matrix import evaluate_obstacle_matrix

        records = evaluate_obstacle_matrix(
            n_per_case=1,
            seed=0,
            cases=[("right_low", [0.40, 0.15, 0.10])],
            radii=[0.07],
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            set(records[0]),
            {"case", "position", "radius", "baseline", "avoidance"},
        )
        self.assertEqual(records[0]["case"], "right_low")
        self.assertEqual(records[0]["baseline"]["n"], 1)
        self.assertEqual(records[0]["avoidance"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
