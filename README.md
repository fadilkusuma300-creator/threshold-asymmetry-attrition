# 🔬 Threshold Asymmetry in Employee Attrition Prediction

> Quantifying how asymmetric classification thresholds inflate reported ensemble advantages in HR attrition model comparisons.

---

## 📋 Overview

In HR attrition prediction, ensemble methods are routinely reported to outperform single classifiers. However, existing comparison studies commonly suffer from **threshold asymmetry**: the proposed ensemble uses an optimized threshold τ*, while baselines are stuck with the default τ = 0.50.

This repository contains the full experimental code for a controlled study that quantifies this methodological blind spot.

## 🎯 Key Findings

- 🏆 Heterogeneous ensemble (RF + XGBoost + LightGBM + SVM) **significantly outperforms** SMOTE+LightGBM under all three threshold conditions
- 📊 Fair-threshold F1 gap: only **0.022** → Asymmetric gap: **0.134** (≈ **6× inflation**)
- ⚠️ ~**84%** of the reported gap is attributable to threshold asymmetry, not model architecture
- 🔍 Threshold optimization is the **single most impactful component** (ΔF1 = −0.089), far exceeding ensemble design (ΔF1 = −0.034) and feature engineering (ΔF1 = −0.012)

## 🧪 Experimental Design

### Three-Condition Controlled Comparison

| Condition | Ensemble Threshold | Baseline Threshold | Purpose |
|-----------|-------------------|-------------------|---------|
| ① Equal Threshold | τ = 0.50 | τ = 0.50 | Raw classification ability |
| ② Fair Threshold | τ* (own optimal) | τ* (own optimal) | Fair architectural advantage |
| ③ Asymmetric Threshold | τ* (only ensemble) | τ = 0.50 | Reproduce common literature practice |

### 📦 Datasets

| Dataset | Samples | Imbalance Ratio | Role |
|---------|---------|----------------|------|
| IBM HR Attrition | 1,470 | 5.2 : 1 | Primary experiment |
| Indian HR Attrition | 5,000 | 1.7 : 1 | Generalization validation |

### 📐 Evaluation Protocol

- **5×5 repeated stratified cross-validation** (25 measurements total)
- **Nadeau-Bengio corrected resampled t-test** (α = 0.05)
- **Metrics**: AUC-ROC, F1, Precision, Recall, G-Mean, Brier Score

## 📁 Project Structure

```
├── config.py                    # Global config (paths, hyperparams, random seed)
├── data_loader.py               # Dataset loading utilities
├── preprocessing.py             # Feature engineering (interactions, scaling, clustering)
├── models.py                    # Model definitions (baselines, ensemble, threshold opt.)
│
├── run_final.py                 # ⭐ Main experiment (3-condition comparison + ablation)
├── run_supplementary_v2.py      # Supplementary baselines (literature methods)
├── plot_figures_revised.py      # Publication-quality figure generation
│
├── data/                        # Datasets
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── hr_attrition_indian.csv
│   └── ziya07_attrition.csv
│
├── results/                     # Output directory
│   └── figures/                 # Paper figures (PDF + PNG + SVG)
│
└── requirements.txt             # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.11

### Installation

```bash
pip install -r requirements.txt
```

### ▶️ Run Experiments

**Main experiment** (three-condition comparison + ablation + calibration):
```bash
python run_final.py --dataset ibm
python run_final.py --dataset indian
```

Supported datasets: `ibm`, `indian`, `ziya07`

**Supplementary baselines** (literature method reproduction):
```bash
python run_supplementary_v2.py --dataset ibm
python run_supplementary_v2.py --dataset indian
```

**Generate paper figures**:
```bash
python plot_figures_revised.py
```

## 🏗️ Method Summary

### Heterogeneous Ensemble

Equal-weight soft voting of four complementary base learners:

| Base Learner | Decision Boundary | Cost-Sensitive Mechanism |
|-------------|-------------------|--------------------------|
| Random Forest | Axis-aligned piecewise | `class_weight='balanced'` |
| XGBoost | Axis-aligned (level-wise) | `scale_pos_weight=IR` |
| LightGBM | Axis-aligned (leaf-wise) | `scale_pos_weight=IR` |
| SVM (RBF) | Smooth curved surface | `class_weight='balanced'` |

### Threshold Optimization

Adaptive search on internal validation set:

$$\tau^* = \arg\max_{\tau} \left[ 0.55 \times F1(\tau) + 0.45 \times G\text{-}Mean(\tau) \right]$$

### Probability Calibration

Isotonic Regression calibration applied to ensemble probability outputs.

## 📄 License

MIT License
