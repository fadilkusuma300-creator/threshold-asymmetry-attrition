"""
数据预处理模块：编码、标准化、特征工程、数据划分。

预处理流程：
  1. 删除零方差 / ID 类无用列
  2. 目标变量编码（Attrition → 0/1）
  3. 分类变量标签编码（LabelEncoder）
  4. 交互特征构造（加班×满意度、收入/工龄、晋升停滞比）
  5. Z-score 标准化
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE


# 需要删除的无用列（零方差或 ID 类）
DROP_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]


def preprocess(df: pd.DataFrame, fit_encoders: dict = None, fit_scaler=None):
    """
    预处理 HR 数据集。

    Parameters
    ----------
    df : DataFrame
        原始数据集。
    fit_encoders : dict or None
        已拟合的标签编码器字典（用于验证集/测试集复用）。
    fit_scaler : StandardScaler or None
        已拟合的标准化器（用于验证集/测试集复用）。

    Returns
    -------
    X : np.ndarray        — 特征矩阵（已标准化）
    y : np.ndarray        — 二分类目标（1 = 离职）
    feature_names : list  — 特征名列表
    encoders : dict       — 拟合后的 LabelEncoder 字典
    scaler : StandardScaler — 拟合后的标准化器
    """
    df = df.copy()

    # --- 删除无用列 ---
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- 目标变量编码 ---
    att_dtype = str(df["Attrition"].dtype)
    if att_dtype in ("object", "string", "str"):
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    y = df["Attrition"].values
    df.drop(columns=["Attrition"], inplace=True)

    # --- 分类变量标签编码 ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = fit_encoders if fit_encoders is not None else {}

    for col in cat_cols:
        if col in encoders:
            # 复用已有编码器，处理未见过的类别
            le = encoders[col]
            df[col] = df[col].map(lambda x, _le=le: _le.transform([x])[0]
                                  if x in _le.classes_ else -1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # --- 交互特征构造 ---
    # 加班状态 × 工作满意度（高负荷 + 低满意度的叠加效应）
    if "OverTime" in df.columns and "JobSatisfaction" in df.columns:
        df["OT_x_JobSat"] = df["OverTime"] * df["JobSatisfaction"]
    # 月收入 / (在职年限 + 1)（单位工龄薪酬增长速度）
    if "MonthlyIncome" in df.columns and "YearsAtCompany" in df.columns:
        df["Income_per_Year"] = df["MonthlyIncome"] / (df["YearsAtCompany"] + 1)
    # 上次晋升距今时间 / (在职年限 + 1)（职业发展停滞程度）
    if "YearsAtCompany" in df.columns and "YearsSinceLastPromotion" in df.columns:
        df["Promo_Gap_Ratio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    feature_names = df.columns.tolist()

    # --- Z-score 标准化 ---
    X = df.values.astype(np.float64)
    if fit_scaler is not None:
        scaler = fit_scaler
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_names, encoders, scaler


def split_data(X, y, test_size=0.2):
    """分层划分训练集/测试集。"""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    from data_loader import load_ibm_hr
    df = load_ibm_hr()
    X, y, feat_names, enc, sc = preprocess(df)
    print(f"特征矩阵: {X.shape}, 目标分布: {np.bincount(y)}")
    print(f"特征数: {len(feat_names)}")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
