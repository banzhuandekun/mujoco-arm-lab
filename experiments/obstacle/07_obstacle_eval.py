"""07 — 障碍物场景下的现有 IK 基线评测。

运行：
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\07_obstacle_eval.py
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\07_obstacle_eval.py --n-eval 100
"""

import argparse

import numpy as np

from mujoco_arm_lab.obstacle import (
    DEFAULT_OBSTACLE_POSITION,
    DEFAULT_OBSTACLE_RADIUS,
    evaluate_ik,
)


def main():
    """运行障碍物场景的 DLS-IK 基线评测并打印逐轮结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--obstacle-radius",
        type=float,
        default=DEFAULT_OBSTACLE_RADIUS,
        help="球形障碍物半径（米）",
    )
    args = parser.parse_args()

    summary, episodes = evaluate_ik(
        n=args.n_eval,
        seed=args.seed,
        obstacle_position=DEFAULT_OBSTACLE_POSITION,
        obstacle_radius=args.obstacle_radius,
    )

    for episode in episodes:
        status = "OK" if episode["success"] else "MISS"
        collision = "COLLISION" if episode["collision"] else "clear"
        print(
            f"[{episode['index'] + 1:02d}] {status:4s} {collision:9s} "
            f"目标={np.round(episode['target'], 3)} "
            f"误差={episode['error'] * 100:5.2f}cm "
            f"耗时={episode['duration']:4.2f}s"
        )

    print("\n--- 障碍物 IK 基线 ---")
    print(f"目标点数 : {summary['n']}")
    print(
        f"成功率   : {summary['success_count']}/{summary['n']} "
        f"= {summary['success_rate'] * 100:.1f}%"
    )
    print(
        f"碰撞率   : {summary['collision_count']}/{summary['n']} "
        f"= {summary['collision_rate'] * 100:.1f}%"
    )
    print(f"平均误差 : {summary['mean_error'] * 100:.2f} cm")
    print(f"平均耗时 : {summary['mean_duration']:.2f} s")
    print(f"超时次数 : {summary['timeout_count']}")


if __name__ == "__main__":
    main()
