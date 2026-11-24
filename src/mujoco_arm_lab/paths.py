"""项目内稳定路径。

模块集中计算模型和输出目录，避免各个入口脚本重复使用 ``__file__`` 拼路径。
这样脚本放在 ``scripts/``、``experiments/`` 或 ``tools/`` 下都能指向同一份资产。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_PATH = MODELS_DIR / "arm3dof.xml"
OBSTACLE_MODEL_PATH = MODELS_DIR / "arm3dof_obstacle.xml"
