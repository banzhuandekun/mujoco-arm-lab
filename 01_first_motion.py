"""
01 — 让机械臂动起来（最小可运行版本）

这一版只做三件事，目的是把整条链路先跑通：
  1. 从 XML 加载模型、创建仿真数据
  2. 用位置执行器驱动三个关节按正弦轨迹摆动
  3. 把画面渲染成 MP4，并存一张 PNG

运行：
    .\\.venv\\Scripts\\python.exe 01_first_motion.py
"""

import os

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "arm3dof.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30
RENDER_EVERY = 16  # 每 16 个仿真步渲染一帧 = 0.032 s，接近实时


def make_camera(model):
    """固定一个好看的观察机位（不跟随机械臂）"""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 1.3
    cam.azimuth = 140
    cam.elevation = -18
    cam.lookat[:] = [0.15, 0.0, 0.30]
    return cam


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print(f"[模型] 自由度 nv={model.nv}，关节数 njnt={model.njnt}，执行器 nu={model.nu}")
    print(f"[模型] 仿真步长 dt={model.opt.timestep}s，物理频率 {1 / model.opt.timestep:.0f} Hz")

    cam = make_camera(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    duration = 4.0
    n_steps = int(duration / model.opt.timestep)
    frames = []

    for step in range(n_steps):
        t = data.time
        # 三个关节各走一条不同频率、不同幅度的正弦轨迹
        data.ctrl[0] = 0.80 * np.sin(2 * np.pi * 0.25 * t)            # 基座左右回转
        data.ctrl[1] = -0.55 + 0.45 * np.sin(2 * np.pi * 0.40 * t)     # 大臂抬起 / 放下
        data.ctrl[2] = 0.85 * np.sin(2 * np.pi * 0.30 * t + 1.0)       # 小臂弯曲 / 伸展
        mujoco.mj_step(model, data)

        if step % RENDER_EVERY == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    video_path = os.path.join(OUT_DIR, "01_first_motion.mp4")
    imageio.mimsave(video_path, frames, fps=FPS, macro_block_size=None)
    imageio.imwrite(os.path.join(OUT_DIR, "01_first_motion.png"), frames[len(frames) // 2])

    print(f"[输出] {len(frames)} 帧 -> {video_path}")
    print(f"[输出] 末端最终位置 = {np.round(data.site_xpos[ee_id], 3)}")


if __name__ == "__main__":
    main()
