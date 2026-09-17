"""
02 — 末端到达目标点：批量评测 + 录制演示

控制逻辑（DLS 逆运动学）在 arm_control.py，这个脚本只负责：
  1. 随机采 n 个目标点，统计到达成功率
  2. 录几段「够到目标点」的连续画面拼成 MP4

场景里绿色小球是目标点（mocap body），橙红色小球是末端（site "ee"）。
误差 < 5 mm 判定为「到达」。

运行：
    .\\.venv\\Scripts\\python.exe 02_reach_target.py
    .\\.venv\\Scripts\\python.exe 02_reach_target.py --n-eval 50 --no-video
"""

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

from arm_control import MODEL_PATH, TOL, Reacher, sample_target

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30


def evaluate(n, seed=0):
    """批量评测：随机采 n 个目标点，统计到达成功率"""
    rng = np.random.default_rng(seed)
    reacher = Reacher()
    dt = reacher.model.opt.timestep

    errors, times = [], []
    for i in range(n):
        target = sample_target(rng)
        err, steps, _ = reacher.run(target)
        errors.append(err)
        times.append(steps * dt)
        mark = "OK  " if err < TOL else "MISS"
        print(f"  [{mark}] {i + 1:2d}  目标 {np.round(target, 3)}  "
              f"误差 {err * 100:5.2f} cm  耗时 {steps * dt:4.2f} s")

    errors = np.array(errors)
    times = np.array(times)
    success = errors < TOL

    print("\n--- 评测结果 ---")
    print(f"  目标点数 : {n}")
    print(f"  成功率   : {success.sum()}/{n} = {success.mean() * 100:.1f}%")
    print(f"  平均误差 : {errors.mean() * 100:.2f} cm")
    if success.any():
        print(f"  成功样本平均耗时: {times[success].mean():.2f} s")
    return success.mean()


def record_video(path, n=3, seed=1):
    """录几段「够到目标点」的连续画面拼成一个 MP4"""
    rng = np.random.default_rng(seed)
    reacher = Reacher()
    reacher.enable_renderer()

    frames = []
    dt = reacher.model.opt.timestep
    for i in range(n):
        target = sample_target(rng)
        err, steps, clip = reacher.run(target, record=True)
        frames.extend(clip)
        print(f"  片段 {i + 1}: 目标 {np.round(target, 3)}  误差 {err * 100:.2f} cm  "
              f"耗时 {steps * dt:.2f} s")

    imageio.mimsave(path, frames, fps=FPS, macro_block_size=None)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=30, help="评测用的随机目标点数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--no-video", action="store_true", help="只评测，不录视频")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== 批量评测：随机目标点到达 ===")
    evaluate(args.n_eval, seed=args.seed)

    if not args.no_video:
        print("\n=== 录制演示视频 ===")
        video_path = os.path.join(OUT_DIR, "02_reach_target.mp4")
        frames = record_video(video_path, n=3, seed=args.seed + 1)
        imageio.imwrite(os.path.join(OUT_DIR, "02_reach_target.png"), frames[-1])
        print(f"[输出] {len(frames)} 帧 -> {video_path}")


if __name__ == "__main__":
    main()
