"""静态球形障碍物的位置与半径评测矩阵。"""

import numpy as np

from .avoidance import evaluate_avoidance
from .obstacle import evaluate_ik

DEFAULT_CASES = (
    ("right_low", [0.40, 0.15, 0.10]),
    ("left_low", [0.40, -0.15, 0.10]),
    ("right_mid", [0.30, 0.15, 0.25]),
    ("left_mid", [0.30, -0.15, 0.25]),
    ("upper_center", [0.22, 0.00, 0.32]),
)
DEFAULT_RADII = (0.05, 0.07)


def evaluate_obstacle_matrix(
    n_per_case,
    seed=0,
    cases=None,
    radii=None,
):
    """对每组障碍物位置和半径，用相同目标点比较两种控制方法。"""
    cases = DEFAULT_CASES if cases is None else cases
    radii = DEFAULT_RADII if radii is None else radii
    records = []

    for case_name, position in cases:
        position = np.asarray(position, dtype=float)
        for radius in radii:
            baseline, _ = evaluate_ik(
                n_per_case,
                seed=seed,
                obstacle_position=position,
                obstacle_radius=radius,
            )
            avoidance, _ = evaluate_avoidance(
                n_per_case,
                seed=seed,
                obstacle_position=position,
                obstacle_radius=radius,
            )
            records.append(
                {
                    "case": case_name,
                    "position": position.copy(),
                    "radius": float(radius),
                    "baseline": baseline,
                    "avoidance": avoidance,
                }
            )

    return records
