"""
Global configuration: project paths, experiment hyperparameters, plot settings.
"""
import os

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Dataset filenames (stored in data/ directory)
# ============================================================
# IBM HR Attrition — primary dataset (1470 samples, IR ≈ 5.2:1)
IBM_HR_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Indian HR Attrition — generalization dataset (5000 samples, IR ≈ 1.7:1)
INDIAN_HR_FILE = os.path.join(DATA_DIR, "hr_attrition_indian.csv")
# Ziya07 Attrition — tertiary dataset (10000 samples)
ZIYA07_FILE = os.path.join(DATA_DIR, "ziya07_attrition.csv")

# ============================================================
# Experiment settings
# ============================================================
RANDOM_STATE = 42       # Global random seed
N_REPEATS = 5           # Number of CV repetitions
N_SPLITS = 5            # Folds per repetition (N_REPEATS × N_SPLITS = 25 measurements)

# K-Means clustering distance features
MAX_K = 6               # Upper bound for cluster count search
MIN_K = 2               # Lower bound for cluster count search

# ============================================================
# Plot settings
# ============================================================
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 12)
DPI = 300
FONT_SIZE = 12
