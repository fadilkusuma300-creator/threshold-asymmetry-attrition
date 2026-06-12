"""
全局配置：项目路径、实验超参数、绘图设置。
"""
import os

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# 数据集文件名（存放在 data/ 目录下）
# ============================================================
# IBM HR Attrition — 主实验数据集（1470 样本，不平衡比率 5.2:1）
IBM_HR_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Indian HR Attrition — 泛化验证数据集（5000 样本，不平衡比率 1.7:1）
INDIAN_HR_FILE = os.path.join(DATA_DIR, "hr_attrition_indian.csv")
# Ziya07 Attrition — 第三数据集（10000 样本）
ZIYA07_FILE = os.path.join(DATA_DIR, "ziya07_attrition.csv")

# ============================================================
# 实验设置
# ============================================================
RANDOM_STATE = 42       # 全局随机种子
N_REPEATS = 5           # 重复交叉验证的重复次数
N_SPLITS = 5            # 每重复的折数（共 N_REPEATS × N_SPLITS = 25 次测量）

# K-Means 聚类距离特征
MAX_K = 6               # 最大聚类数搜索上限
MIN_K = 2               # 最小聚类数

# ============================================================
# 绘图设置
# ============================================================
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 12)
DPI = 300
FONT_SIZE = 12
