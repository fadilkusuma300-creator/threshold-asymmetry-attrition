# Threshold Asymmetry in Employee Attrition Prediction

本仓库包含论文《员工离职预测模型比较中阈值非对称性的影响与量化评估》的全部实验代码。

## 研究问题

在 HR 离职预测的模型比较研究中，集成方法常被报告为优于单一分类器。然而，现有比较实验普遍存在**阈值条件非对称**的问题：集成方法使用优化阈值 τ*，而基线方法使用默认阈值 0.50。本文通过三组受控对照实验，量化了这种非对称性对模型比较结论的扭曲程度。

## 核心发现

- 异构集成（RF + XGBoost + LightGBM + SVM）在三种阈值条件下均显著优于 SMOTE+LightGBM
- 公平阈值条件下 F1 差距仅 0.022，非对称条件下扩大至 0.134（放大约 6 倍）
- 约 84% 的报告差距可归因于阈值非对称设置，而非模型架构本身
- 阈值优化是消融实验中贡献最大的单一组件（ΔF1 = −0.089），远超异构集成设计（ΔF1 = −0.034）和特征工程（ΔF1 = −0.012）

## 实验设计

### 三条件对照实验

| 条件 | 集成方法阈值 | 基线方法阈值 | 检验目的 |
|------|-------------|-------------|---------|
| 条件一（等阈值） | τ = 0.50 | τ = 0.50 | 原生分类能力差异 |
| 条件二（公平阈值） | τ*（各自最优） | τ*（各自最优） | 公平条件下的架构优势 |
| 条件三（非对称阈值） | τ*（仅集成优化） | τ = 0.50 | 复现文献中的常见做法 |

### 数据集

| 数据集 | 样本数 | 不平衡比率 | 用途 |
|--------|--------|-----------|------|
| IBM HR Attrition | 1,470 | 5.2:1 | 主实验 |
| Indian HR Attrition | 5,000 | 1.7:1 | 泛化验证 |

### 评估协议

- 5×5 重复分层交叉验证（共 25 次测量）
- Nadeau-Bengio 修正重采样 t 检验（α = 0.05）
- 评价指标：AUC-ROC、F1、Precision、Recall、G-Mean、Brier Score

## 项目结构

```
├── config.py                    # 全局配置（路径、超参、随机种子）
├── data_loader.py               # 数据集加载与缓存
├── preprocessing.py             # 特征工程（交互特征、标准化、聚类距离）
├── models.py                    # 模型定义（基线方法、异构集成、阈值优化）
│
├── run_final.py                 # 主实验脚本（三条件对照 + 消融 + 校准对比）
├── run_supplementary_v2.py      # 补充基线（文献方法复现）
├── plot_figures_revised.py      # 论文图表生成
│
├── data/                        # 数据集
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── hr_attrition_indian.csv
│   └── ziya07_attrition.csv
│
├── results/                     # 实验结果输出目录
│   └── figures/                 # 论文配图
│
└── requirements.txt             # Python 依赖
```

## 环境要求

- Python >= 3.11
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 复现实验

### 主实验（三条件对照 + 消融）

```bash
python run_final.py --dataset ibm
python run_final.py --dataset indian
```

支持的数据集参数：`ibm`、`indian`、`ziya07`

### 补充基线实验

```bash
python run_supplementary_v2.py --dataset ibm
python run_supplementary_v2.py --dataset indian
```

### 生成论文图表

```bash
python plot_figures_revised.py
```

## 方法概述

### 异构集成

由四种基学习器组成的等权软投票集成：
- **Random Forest**：Bootstrap 采样 + 随机特征子集
- **XGBoost**：按层生长梯度提升
- **LightGBM**：按叶生长梯度提升
- **SVM**：RBF 核映射（与树模型决策边界互补）

所有基学习器均采用代价敏感设置（class_weight / scale_pos_weight）。

### 阈值优化

在内部验证集上搜索最优分类阈值：

$$\tau^* = \arg\max_{\tau} \left[ 0.55 \times F1(\tau) + 0.45 \times G\text{-}Mean(\tau) \right]$$

### 概率校准

对集成概率输出应用等距回归（Isotonic Regression）校准。

## 许可证

MIT License
