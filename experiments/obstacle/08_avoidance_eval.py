"""08 — 同一批目标点上的避障前后对比。

运行：
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\08_avoidance_eval.py
    .\\.venv\\Scripts\\python.exe experiments\\obstacle\\08_avoidance_eval.py --n-eval 200
"""

import argparse

from mujoco_arm_lab.avoidance import evaluate_avoidance
from mujoco_arm_lab.obstacle import evaluate_ik


def format_row(name, summary):
    """把一组汇总指标格式化为对比表的一行。"""
    return (
        f"{name:<14}"
        f"{summary['success_count']:>3}/{summary['n']:<5}"
        f"{summary['success_rate'] * 100:>9.1f}%"
        f"{summary['collision_count']:>10}"
        f"{summary['collision_rate'] * 100:>9.1f}%"
        f"{summary['mean_error'] * 100:>12.2f} cm"
        f"{summary['mean_duration']:>11.2f} s"
        f"{summary['timeout_count']:>8}"
    )


def main():
    """在同一批目标点上比较原始 DLS-IK 与避障策略。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"评测 {args.n_eval} 个相同目标点，随机种子 {args.seed}")
    print("判据：末端误差 < 5 mm，单目标最多 8 s 仿真时间\n")

    baseline, _ = evaluate_ik(args.n_eval, seed=args.seed)
    avoidance, _ = evaluate_avoidance(args.n_eval, seed=args.seed)

    print(
        f"{'方法':<14}{'成功':<9}{'成功率':>10}{'碰撞':>10}"
        f"{'碰撞率':>10}{'平均误差':>15}{'平均耗时':>13}{'超时':>8}"
    )
    print("-" * 91)
    print(format_row("DLS-IK", baseline))
    print(format_row("避障策略", avoidance))


if __name__ == "__main__":
    main()
