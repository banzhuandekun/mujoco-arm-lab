"""
04 — 用 PPO 训练机械臂到达目标点

目的不是「用 RL 替代 IK」，而是拿到一个可以对比的基线：
同一批随机目标点、同一时间预算、同一到达判据，看两种方法各有什么长短。

训练规模参考（自己试出来的经验值）：
  --timesteps 50000    快速试跑，几分钟出结果，能看出学没学会
  --timesteps 200000   默认，效果比较稳

运行：
    .\\.venv\\Scripts\\python.exe 04_train_ppo.py
    .\\.venv\\Scripts\\python.exe 04_train_ppo.py --timesteps 50000
"""

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from arm_env import ArmReachEnv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "ppo_arm.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000, help="总训练步数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2048, help="每次更新采多少步")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--name", type=str, default="ppo_arm", help="模型文件名（不含扩展名）")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    env = DummyVecEnv([lambda: ArmReachEnv(seed=args.seed)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        verbose=1,
        seed=args.seed,
        device="cpu",          # 这台机器上 torch 是 CPU 版，显式指定避免警告
    )

    print(f"开始训练：{args.timesteps} 步，CPU，可能需要几分钟……\n")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0

    path = os.path.join(OUT_DIR, f"{args.name}.zip")
    model.save(path)
    print(f"\n训练完成：{args.timesteps} 步，用时 {elapsed / 60:.1f} 分钟"
          f"（{args.timesteps / elapsed:.0f} 步/秒）")
    print(f"模型已保存到 {path}")
    print("\n下一步：跑 05_compare.py 看它和 IK 的对比结果")


if __name__ == "__main__":
    main()
