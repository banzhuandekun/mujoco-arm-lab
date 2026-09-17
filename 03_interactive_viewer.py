"""
03 — 交互式查看器：实时看着机械臂去够随机目标点

和前两个脚本不同，这个会弹出一个真实的 3D 窗口：
  - 鼠标左键拖动：旋转视角
  - 鼠标右键拖动：平移
  - 滚轮：缩放
  - 关闭窗口或按 ESC：退出

每够到一个目标点，停 0.5 秒后自动换下一个，控制台打印误差。

注意一个和 02 不同的地方：这里连续作业、不重置机械臂姿态，
个别目标会陷在局部极小或贴住关节限位，怎么都够不到。
所以这里设了单目标超时（8 s 仿真时间），超时就换下一个。

运行：
    .\\.venv\\Scripts\\python.exe 03_interactive_viewer.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from arm_control import IK_EVERY, TOL, Reacher, sample_target

HOLD_STEPS = 250            # 够到目标后停 250 步 = 0.5 s，再换下一个目标
TARGET_TIMEOUT_STEPS = 4000  # 单个目标最多尝试 8 s 仿真时间，超时就换下一个


def main():
    rng = np.random.default_rng()
    reacher = Reacher()
    model, data = reacher.model, reacher.data
    dt = model.opt.timestep

    target = sample_target(rng)
    reacher.reset(target)

    step = 0
    hold = 0
    attempt_steps = 0
    reached = 0
    missed = 0
    reached_flag = False  # 是否已经够到当前目标（锁存，避免误差在阈值附近抖动导致反复触发）

    print("窗口已打开：左键拖动转视角，滚轮缩放，关窗退出。")
    print("机械臂会自动寻找一个又一个随机目标点。\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -18
        viewer.cam.lookat[:] = [0.15, 0.0, 0.30]

        while viewer.is_running():
            t0 = time.time()

            # 到达判定放在 IK 周期上，和 02 的评测口径保持一致
            if step % IK_EVERY == 0 and not reached_flag:
                err = reacher.error_to(target)
                if err < TOL:
                    reached_flag = True
                    reached += 1
                    print(f"[{reached:3d}] 到达  目标 {np.round(target, 3)}  "
                          f"误差 {err * 100:.2f} cm  用时 {attempt_steps * dt:.1f} s")
                else:
                    reacher.ik_step(target)

            mujoco.mj_step(model, data)
            step += 1
            attempt_steps += 1

            if reached_flag:
                # 够到之后保持 0.5 s，再换下一个目标点
                hold += 1
                if hold > HOLD_STEPS:
                    hold = 0
                    reached_flag = False
                    attempt_steps = 0
                    target = sample_target(rng)
                    reacher.set_target(target)
            elif attempt_steps >= TARGET_TIMEOUT_STEPS:
                # 连续作业不重置姿态，个别目标会卡在局部极小 / 关节限位附近。
                # 这不是控制器坏了，而是「逐轮重置」和「连续作业」两种工况的真实差别。
                missed += 1
                err = reacher.error_to(target)
                print(f"[  -] 超时  目标 {np.round(target, 3)}  "
                      f"误差 {err * 100:.2f} cm -> 换下一个")
                attempt_steps = 0
                target = sample_target(rng)
                reacher.set_target(target)

            viewer.sync()
            # 让仿真按真实时间推进（不加这句会以 CPU 极限速度狂飙）
            time.sleep(max(0.0, dt - (time.time() - t0)))

    print(f"\n结束：到达 {reached} 个，超时 {missed} 个 "
          f"(连续作业模式成功率 {reached / max(1, reached + missed) * 100:.0f}%)")


if __name__ == "__main__":
    main()
