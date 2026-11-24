"""09 — 障碍物位置与半径矩阵评测。

运行：
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\09_obstacle_matrix.py
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\09_obstacle_matrix.py --n-eval 100
"""

import argparse

from mujoco_arm_lab.matrix import DEFAULT_RADII, evaluate_obstacle_matrix


def format_summary(summary):
    """把基线或避障结果压缩成矩阵表所需的一列。"""
    return (
        f"{summary['success_count']:>3}/{summary['n']:<4}"
        f"{summary['collision_count']:>5}/{summary['n']:<4}"
        f"{summary['mean_duration']:>8.2f}s"
        f"{summary['timeout_count']:>7}"
    )


def main():
    """运行位置 × 半径矩阵，并打印两种控制器的对照结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = evaluate_obstacle_matrix(args.n_eval, seed=args.seed)

    print(
        f"每种障碍物配置评测 {args.n_eval} 个相同目标点，随机种子 {args.seed}"
    )
    print("成功/碰撞均为次数；耗时为每个目标的平均仿真时间\n")
    print(
        f"{'位置':<15}{'半径':<8}{'基线成功':<11}{'基线碰撞':<11}"
        f"{'基线耗时':<11}{'超时':<7}{'避障成功':<11}"
        f"{'避障碰撞':<11}{'避障耗时':<11}{'超时':<7}"
    )
    print("-" * 108)
    for record in records:
        baseline = record["baseline"]
        avoidance = record["avoidance"]
        print(
            f"{record['case']:<15}"
            f"{record['radius'] * 100:<6.0f}cm"
            f"{format_summary(baseline):<40}"
            f"{format_summary(avoidance):<40}"
        )

    print("\n默认半径：" + ", ".join(f"{radius * 100:.0f}cm" for radius in DEFAULT_RADII))


if __name__ == "__main__":
    main()
