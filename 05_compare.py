"""
05 — 对比评测：IK（解析解） vs BC（学出来的策略）

严格对齐四件事，否则数字没有可比性：
  1. 同一批随机目标点（同一个随机种子）
  2. 同一时间预算（单目标最多 8 s 仿真时间）
  3. 同一到达判据（末端误差 < 5 mm）
  4. 同一控制频率（每 5 个仿真步给一次指令）

除了成功率，还测「算一次控制指令要多久」—— 这是两者本质的区别：
IK 每步都要解雅可比方程，BC 只是一次神经网络前向。

运行：
    .\\.venv\\Scripts\\python.exe 05_compare.py
    .\\.venv\\Scripts\\python.exe 05_compare.py --n-eval 100
"""

import argparse
import os
import time

import mujoco
import numpy as np

from arm_bc import BCPolicy
from arm_control import LAMBDA, TOL, Reacher, sample_target
from arm_env import ACTION_REPEAT, ArmReachEnv

ROOT = os.path.dirname(os.path.abspath(__file__))
BC_MODEL = os.path.join(ROOT, "outputs", "bc_policy.pt")
TIME_BUDGET = 8.0   # s，单目标的时间上限，和 02 里 IK 的 MAX_STEPS 一致


def make_targets(n, seed):
    """用固定种子生成目标点，保证两种方法面对的是完全一样的题"""
    rng = np.random.default_rng(seed)
    return [sample_target(rng) for _ in range(n)]


def eval_ik(targets):
    reacher = Reacher()
    dt = reacher.model.opt.timestep
    errors, durations = [], []
    for target in targets:
        err, steps, _ = reacher.run(target)
        errors.append(err)
        durations.append(steps * dt)
    return np.array(errors), np.array(durations)


def eval_policy(policy, targets):
    """用环境接口跑策略（BC 或任何有 predict() 的对象）"""
    env = ArmReachEnv()
    dt = env.model.opt.timestep
    errors, durations = [], []
    for target in targets:
        obs, _ = env.reset(options={"target": target})
        decisions = 0
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            decisions += 1
            done = terminated or truncated
        errors.append(env.reacher.error_to(env.target))
        durations.append(decisions * ACTION_REPEAT * dt)
    return np.array(errors), np.array(durations)


def time_ik_step(n=2000):
    """测 IK 解一次逆运动学要多久（微秒）"""
    reacher = Reacher()
    jacp = np.zeros((3, reacher.model.nv))
    err = np.array([0.05, 0.03, -0.02])
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_jacSite(reacher.model, reacher.data, jacp, None, reacher.site_id)
        A = jacp @ jacp.T + (LAMBDA ** 2) * np.eye(3)
        jacp.T @ np.linalg.solve(A, err)
    return (time.perf_counter() - t0) / n * 1e6


def time_policy_step(policy, n=2000):
    """测 BC 网络前向一次要多久（微秒）"""
    obs = np.zeros(10, dtype=np.float32)
    policy.predict(obs)          # 预热，别把第一次的初始化开销算进去
    t0 = time.perf_counter()
    for _ in range(n):
        policy.predict(obs)
    return (time.perf_counter() - t0) / n * 1e6


def make_row(name, errors, durations, step_us):
    success = errors < TOL
    return {
        "method": name,
        "success": "{} / {} = {:.1f}%".format(success.sum(), len(errors), success.mean() * 100),
        "mean_err": "{:.2f} cm".format(errors.mean() * 100),
        "worst_err": "{:.2f} cm".format(errors.max() * 100),
        "mean_time": "{:.2f} s".format(durations.mean()),
        "timeouts": str(int((durations >= TIME_BUDGET - 1e-9).sum())),
        "step_us": "{:.1f} us".format(step_us),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(BC_MODEL):
        raise SystemExit("找不到 BC 模型 {}，先跑 06_train_bc.py 训练一个".format(BC_MODEL))

    targets = make_targets(args.n_eval, args.seed)
    print("评测 {} 个随机目标点（随机种子 {}）".format(args.n_eval, args.seed))
    print("判据：末端误差 < {:.0f} mm，单目标最多 {:.0f} s 仿真时间\n".format(TOL * 1000, TIME_BUDGET))

    print("跑 IK ……")
    ik_err, ik_time = eval_ik(targets)
    ik_step_us = time_ik_step()

    print("跑 BC ……")
    bc_policy = BCPolicy.load(BC_MODEL)
    bc_err, bc_time = eval_policy(bc_policy, targets)
    bc_step_us = time_policy_step(bc_policy)

    rows = [make_row("IK (DLS)", ik_err, ik_time, ik_step_us),
            make_row("BC (神经网络)", bc_err, bc_time, bc_step_us)]

    print("\n" + "=" * 90)
    print("对比结果")
    print("=" * 90)
    print("{:<16}{:<20}{:<12}{:<12}{:<12}{:<8}{}".format(
        "方法", "成功率", "平均误差", "最差误差", "平均耗时", "超时", "单步控制"))
    print("-" * 90)
    for r in rows:
        print("{:<16}{:<20}{:<12}{:<12}{:<12}{:<8}{}".format(
            r["method"], r["success"], r["mean_err"], r["worst_err"],
            r["mean_time"], r["timeouts"], r["step_us"]))
    print("=" * 90)


if __name__ == "__main__":
    main()
