"""
把 outputs 里的演示视频转成 GIF。

为什么需要这个：GitHub 的 README 不能直接播放仓库里的 mp4，
想让访客一打开仓库就看到机械臂在动，得用 GIF。

运行：
    .\\.venv\\Scripts\\python.exe tools\\make_gif.py

ffmpeg 由 imageio-ffmpeg 自带，不需要另外安装。
"""

import subprocess

import imageio_ffmpeg

from mujoco_arm_lab.paths import OUTPUTS_DIR

# (源视频, 目标 GIF, 帧率, 宽度)
JOBS = [
    ("02_reach_target.mp4", "02_reach_target.gif", 12, 480),
    ("01_first_motion.mp4", "01_first_motion.gif", 12, 480),
]


def convert(ffmpeg, src, dst, fps, width):
    """单条命令版的调色板法，比直接转出来的 GIF 干净很多"""
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=4"
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-vf", vf, dst],
        check=True,
    )


def main():
    """把已存在的演示视频批量转换成 README 可播放的 GIF。"""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for src_name, dst_name, fps, width in JOBS:
        src = OUTPUTS_DIR / src_name
        dst = OUTPUTS_DIR / dst_name
        if not src.exists():
            print(f"[跳过] 找不到 {src_name}，请先跑对应的脚本")
            continue
        convert(ffmpeg, src, dst, fps, width)
        print(f"[输出] {dst_name}  {dst.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
