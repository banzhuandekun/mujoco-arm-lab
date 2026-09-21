"""
机械臂控制核心 —— 目标点采样 + 阻尼最小二乘逆运动学（DLS-IK）。

01 / 02 / 03 三个脚本共用这里的 Reacher，避免同一套控制逻辑抄三遍。

下面这些参数是实测扫出来的，两条结论都反直觉，改之前先看一眼：
  - 单步增量要「小」：MAX_DQ 从 0.01 放大到 0.03，成功率从 100% 掉到 92%，
    因为位置执行器跟不上跳变的指令，反而来回振荡。
  - 解 IK 要「慢」：IK_EVERY 从 5 改成 1（每步都解），成功率暴跌到 42%，
    同样是「指令跑在伺服前面」导致超调。
"""

import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "arm3dof.xml")


def load_model(path=MODEL_PATH):
    """读取 MJCF 并建模型。

    不用 from_xml_path：MuJoCo 的 C++ 解析器在 Windows 上处理不了含非 ASCII
    字符的路径（项目放在「桌面缓存\乱七八糟\项目」这类中文目录下会直接报
    Error opening file），改由 Python 读文件内容再交给 from_xml_string。
    """
    with open(path, "rb") as f:
        return mujoco.MjModel.from_xml_string(f.read())

IK_EVERY = 5        # 每 5 个仿真步解一次 IK（相当于 100 Hz 控制频率）
MAX_DQ = 0.01       # 单次 IK 每个关节最多走 0.01 rad
LAMBDA = 0.05       # 阻尼系数，防止奇异位形附近解爆掉
TOL = 0.005         # 到达阈值：5 mm
MAX_STEPS = 4000    # 单次尝试最长 8 s 仿真时间
RENDER_EVERY = 16   # 录制时每 16 个仿真步取一帧

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
        self.model = load_model()
        self.data = mujoco.MjData(self.model)

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.model.body_mocapid[target_body]

        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("yaw", "shoulder", "elbow")
        ]
        self.joint_ids = joint_ids
        # q_idx：三个关节角在 qpos 里的下标；执行器顺序和它们一致
        self.q_idx = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        # d_idx：三个关节角速度在 qvel 里的下标（强化学习环境要观测速度）
        self.d_idx = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self.q_lo = self.model.jnt_range[joint_ids, 0].copy()
        self.q_hi = self.model.jnt_range[joint_ids, 1].copy()

        self.jacp = np.zeros((3, self.model.nv))
        self.q_target = None      # 当前下发给执行器的关节目标角
        self.renderer = None
        self.camera = None

    # ---------- 状态查询 ----------

    @property
    def ee_pos(self):
        """末端当前世界坐标"""
        return self.data.site_xpos[self.site_id].copy()

    def error_to(self, target):
        """末端到目标点的距离（米）"""
        return float(np.linalg.norm(target - self.ee_pos))

    # ---------- 初始化 ----------

    def enable_renderer(self, height=480, width=640):
        """需要出视频时才开渲染器（纯统计评测不开能跑更快）"""
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.distance = 1.3
        cam.azimuth = 140
        cam.elevation = -18
        cam.lookat[:] = [0.15, 0.0, 0.30]
        self.camera = cam

    def reset(self, target):
        """回到初始姿态，并把目标点放到 target"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = target
        self.data.ctrl[:] = self.data.qpos[self.q_idx]  # 平稳起步，避免第一帧猛跳
        mujoco.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[self.q_idx].copy()
        return self.q_target.copy()

    def set_target(self, target):
        """只挪动目标点，不重置机械臂（交互式查看器用）"""
        self.data.mocap_pos[self.mocap_id] = target

    # ---------- 控制 ----------

    def solve_dls(self, err):
        """用当前雅可比把末端误差映射成关节增量（阻尼最小二乘）"""
        J = self.jacp
        A = J @ J.T + (LAMBDA ** 2) * np.eye(3)
        return J.T @ np.linalg.solve(A, err)

    def ik_step(self, target):
        """解一次 IK，把新的关节目标角写进 ctrl（调用前请自行判断是否还需要更新）"""
        mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.site_id)
        dq = np.clip(self.solve_dls(target - self.ee_pos), -MAX_DQ, MAX_DQ)
        self.q_target = np.clip(self.q_target + dq, self.q_lo, self.q_hi)
        self.data.ctrl[:] = self.q_target

    def run(self, target, record=False, max_steps=MAX_STEPS):
        """从初始姿态出发控制末端走向 target，返回 (最终误差, 步数, 视频帧)"""
        self.reset(target)
        frames = []
        steps = 0

        for step in range(max_steps):
            # 注意：到达判定只在 IK 周期上做（每 IK_EVERY 步一次）。
            # 如果改成每步都判定，会在误差刚跌到阈值时就提前中止，
            # 实测平均误差会从 0.34 cm 变成 0.46 cm —— 看起来更差，
            # 其实是判定时机不同，不是控制器变弱了。
            if step % IK_EVERY == 0:
                if self.error_to(target) < TOL:
                    break
                self.ik_step(target)

            mujoco.mj_step(self.model, self.data)
            steps += 1

            if record and step % RENDER_EVERY == 0:
                self.renderer.update_scene(self.data, camera=self.camera)
                frames.append(self.renderer.render().copy())

        return self.error_to(target), steps, frames
