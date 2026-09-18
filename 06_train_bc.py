"""
06 — 行为克隆：让神经网络模仿 IK 专家的动作

流程：
  1. 用 02 里那套 DLS 逆运动学当专家，跑随机目标点，记录 (观测, 动作) 对
  2. 采集时以一定概率给机械臂注入随机扰动，逼专家演示「如何从偏离状态纠正回来」
  3. 用监督学习（MSE）训一个 MLP 去拟合这些动作
  4. 评测

第 2 步是关键：不加扰动只能到 62%，加了扰动能到 98%。
原因见 arm_bc.py 里的说明。

运行：
    .\\.venv\\Scripts\\python.exe 06_train_bc.py
    .\\.venv\\Scripts\\python.exe 06_train_bc.py --episodes 150 --epochs 90   # 快速试跑
"""

import argparse
import os
import time

import mujoco
import numpy as np
import torch
import torch.nn as nn

from arm_bc import BCPolicy, BCPolicyNet
from arm_control import LAMBDA, MAX_DQ, TOL, sample_target
from arm_env import ACTION_REPEAT, MAX_ACTION_DQ, ArmReachEnv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "bc_policy.pt")


def ik_action(env, target):
    """专家动作：阻尼最小二乘逆运动学，结果换算到环境的动作尺度"""
    err = target - env.reacher.ee_pos
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None, env.reacher.site_id)
    A = jacp @ jacp.T + (LAMBDA ** 2) * np.eye(3)
    dq = np.clip(jacp.T @ np.linalg.solve(A, err), -MAX_DQ, MAX_DQ)
    return dq / MAX_ACTION_DQ


def collect(episodes, seed, perturb_prob=0.15, perturb_std=0.08):
    """跑专家轨迹并采集数据；perturb_prob 控制注入扰动的频率"""
    env = ArmReachEnv()
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []

    for _ in range(episodes):
        target = sample_target(rng)
        env.reset(options={"target": target})
        done = False
        while not done:
            # 关键：把机械臂推开，让专家演示纠正动作，对抗复合误差
            if perturb_prob > 0 and rng.random() < perturb_prob:
                q = env.reacher.q_target + rng.normal(0, perturb_std, size=3)
                env.reacher.q_target = np.clip(q, env.reacher.q_lo, env.reacher.q_hi)
                env.data.ctrl[:] = env.reacher.q_target
                for _ in range(ACTION_REPEAT):
                    mujoco.mj_step(env.model, env.data)

            obs = env._obs()
            action = ik_action(env, target)
            obs_list.append(obs.copy())
            act_list.append(action)

            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.array(act_list), dtype=torch.float32)
    return X, Y


def train(X, Y, epochs, batch_size=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = BCPolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
    loss_fn = nn.MSELoss()
    n = len(X)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(X[idx]), Y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if (epoch + 1) % max(1, epochs // 3) == 0:
            print(f"    epoch {epoch + 1:3d}  训练 MSE {total / n:.3e}")

    return net


def evaluate(policy, n_eval=50, seed=0):
    env = ArmReachEnv()
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(n_eval):
        target = sample_target(rng)
        obs, _ = env.reset(options={"target": target})
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        errors.append(env.reacher.error_to(env.target))
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300, help="专家轨迹条数")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== 1/3 采集专家数据（含扰动）===")
    t0 = time.time()
    X, Y = collect(args.episodes, seed=2026)
    print(f"  {len(X)} 条样本，用时 {time.time() - t0:.1f} s")

    print("\n=== 2/3 训练行为克隆网络 ===")
    t0 = time.time()
    net = train(X, Y, args.epochs, args.batch_size, seed=args.seed)
    print(f"  训练用时 {time.time() - t0:.1f} s")

    torch.save(net.state_dict(), MODEL_PATH)
    print(f"  网络已保存到 {MODEL_PATH}")

    print(f"\n=== 3/3 评测（{args.n_eval} 个随机目标点）===")
    errors = evaluate(BCPolicy(net), n_eval=args.n_eval, seed=args.seed)
    success = errors < TOL
    print(f"  成功率   : {success.sum()}/{len(errors)} = {success.mean() * 100:.1f}%")
    print(f"  平均误差 : {errors.mean() * 100:.2f} cm")
    print(f"  最差误差 : {errors.max() * 100:.2f} cm")
    print("\n下一步：跑 05_compare.py 看它和 IK 的对比")


if __name__ == "__main__":
    main()
