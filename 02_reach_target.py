"""
02 — 末端到达目标点：阻尼最小二乘逆运动学（DLS-IK）

场景里绿色小球是目标点（mocap body，可瞬移到任意位置），
橙红色小球是末端（site "ee"）。

每个控制周期做四件事：
  1. 读取末端实际位置，算误差 err = target - ee
  2. 用 mj_jacSite 取末端位置雅可比 J（3 x nv）——它描述「关节动一点，末端往哪走」
  3. 解 dq = J^T (J J^T + λ²I)^-1 err，并限制单步幅度
  4. 把 dq 累加到关节目标角，写进 ctrl 交给位置执行器跟踪

误差 < 5 mm 判定为「到达」。

参数是实测调出来的，两个反直觉的结论写在下面，改参数前先看一眼：
  - 单步增量要「小」：MAX_DQ 从 0.01 放大到 0.03，成功率从 100% 掉到 92%，
    因为位置执行器跟不上跳变的指令，反而来回振荡。
  - 解 IK 要「慢」：把 IK_EVERY 从 5 改成 1（每步都解），成功率暴跌到 42%，
    同样是「指令跑在伺服前面」导致超调。

运行：
    .\\.venv\\Scripts\\python.exe 02_reach_target.py
    .\\.venv\\Scripts\\python.exe 02_reach_target.py --n-eval 50 --no-video
"""

import argparse
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "arm3dof.xml")
OUT_DIR = os.path.join(ROOT, "outputs")

FPS = 30
RENDER_EVERY = 16   # 录制时每 16 步取一帧
IK_EVERY = 5        # 每 5 个仿真步解一次 IK，相当于 100 Hz 控制频率
MAX_DQ = 0.01       # 单次 IK 每个关节最多走 0.01 rad（小步慢走，别贪快）
LAMBDA = 0.05       # 阻尼系数，防止奇异位形附近解爆掉
TOL = 0.005         # 到达阈值：5 mm
MAX_STEPS = 4000    # 单次尝试最长 8 s 仿真时间（接近全伸展的目标点需要更久）

# 大臂根部在世界坐标里的位置，用来判断目标点是否可达
SHOULDER_PIVOT = np.array([0.0, 0.0, 0.10])


def sample_target(rng, min_reach=0.18, max_reach=0.66):
    """在可达的空心球壳内随机采一个目标点（世界坐标）"""
    while True:
        r = rng.uniform(0.20, 0.60)
        theta = rng.uniform(-np.pi, np.pi)
        z = rng.uniform(0.12, 0.60)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        if min_reach <= np.linalg.norm(p - SHOULDER_PIVOT) <= max_reach:
            return p


class Reacher:
    """三自由度机械臂 + DLS 逆运动学控制器"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.model.body_mocapid[target_body]

        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("yaw", "shoulder", "elbow")
        ]
        # q_idx：三个关节角在 qpos 里的下标；执行器顺序和它们一致
        self.q_idx = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self.q_lo = self.model.jnt_range[joint_ids, 0].copy()
        self.q_hi = self.model.jnt_range[joint_ids, 1].copy()

        self.jacp = np.zeros((3, self.model.nv))
        self.renderer = None
        self.camera = None

    @property
    def ee_pos(self):
        """末端当前世界坐标"""
        return self.data.site_xpos[self.site_id].copy()

    def enable_renderer(self, height=480, width=640):
        """需要出视频时再开渲染器（纯统计评测时不开能跑更快）"""
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.distance = 1.3
        cam.azimuth = 140
        cam.elevation = -18
        cam.lookat[:] = [0.15, 0.0, 0.30]
        self.camera = cam

    def reset(self, target):
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = target
        self.data.ctrl[:] = self.data.qpos[self.q_idx]  # 从当前姿态平稳起步，避免第一帧猛跳
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[self.q_idx].copy()

    def solve_dls(self):
        """用当前雅可比把末端误差映射成关节增量（阻尼最小二乘）"""
        err = self._err
        J = self.jacp
        A = J @ J.T + (LAMBDA ** 2) * np.eye(3)
        return J.T @ np.linalg.solve(A, err)

    def run(self, target, record=False):
        """控制末端走向 target，返回 (最终误差, 仿真步数, 视频帧)"""
        q_target = self.reset(target)
        frames = []
        steps = 0

        for step in range(MAX_STEPS):
            if step % IK_EVERY == 0:
                self._err = target - self.ee_pos
                if np.linalg.norm(self._err) < TOL:
                    break
                mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.site_id)
                dq = np.clip(self.solve_dls(), -MAX_DQ, MAX_DQ)
                q_target = np.clip(q_target + dq, self.q_lo, self.q_hi)
                self.data.ctrl[:] = q_target

            mujoco.mj_step(self.model, self.data)
            steps += 1

            if record and step % RENDER_EVERY == 0:
                self.renderer.update_scene(self.data, camera=self.camera)
                frames.append(self.renderer.render().copy())

        return float(np.linalg.norm(target - self.ee_pos)), steps, frames


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
        mark = "OK " if err < TOL else "MISS"
        print(f"  [{mark}] {i + 1:2d}  目标 {np.round(target, 3)}  "
              f"误差 {err * 100:5.2f} cm  耗时 {steps * dt:4.2f} s")

    errors = np.array(errors)
    times = np.array(times)
    success = errors < TOL

    print("\n--- 评测结果 ---")
    print(f"  目标点数   : {n}")
    print(f"  成功率     : {success.sum()}/{n} = {success.mean() * 100:.1f}%")
    print(f"  平均误差   : {errors.mean() * 100:.2f} cm")
    print(f"  成功样本平均耗时: {times[success].mean():.2f} s" if success.any() else "")
    return success.mean()


def record_video(path, n=3, seed=1):
    """录几段「够到目标点」的连续画面拼成一个 MP4"""
    rng = np.random.default_rng(seed)
    reacher = Reacher()
    reacher.enable_renderer()

    frames = []
    for i in range(n):
        target = sample_target(rng)
        err, steps, clip = reacher.run(target, record=True)
        frames.extend(clip)
        print(f"  片段 {i + 1}: 目标 {np.round(target, 3)}  误差 {err * 100:.2f} cm  "
              f"耗时 {steps * reacher.model.opt.timestep:.2f} s")

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
