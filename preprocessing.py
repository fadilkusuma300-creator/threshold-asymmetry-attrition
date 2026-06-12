"""
Data preprocessing module: encoding, scaling, feature engineering, train/test split.

Pipeline:
  1. Drop zero-variance / ID-like columns
  2. Encode target variable (Attrition → 0/1)
  3. Label-encode categorical features
  4. Construct interaction features (OT×JobSat, Income/Year, PromoGap)
  5. Z-score standardization
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE


# Columns to drop (zero-variance or ID-like)
DROP_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]


def preprocess(df: pd.DataFrame, fit_encoders: dict = None, fit_scaler=None):
    """
    Preprocess the HR dataset.

    Parameters
    ----------
    df : DataFrame
        Raw dataset.
    fit_encoders : dict or None
        Pre-fitted label encoders (for validation/test set reuse).
    fit_scaler : StandardScaler or None
        Pre-fitted scaler (for validation/test set reuse).

    Returns
    -------
    X : np.ndarray        — feature matrix (scaled)
    y : np.ndarray        — binary target (1 = Attrition)
    feature_names : list  — feature name list
    encoders : dict       — fitted LabelEncoder dict
    scaler : StandardScaler — fitted scaler
    """
    df = df.copy()

    # --- Drop useless columns ---
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- Encode target ---
    att_dtype = str(df["Attrition"].dtype)
    if att_dtype in ("object", "string", "str"):
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    y = df["Attrition"].values
    df.drop(columns=["Attrition"], inplace=True)

    # --- Label-encode categorical features ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = fit_encoders if fit_encoders is not None else {}

    for col in cat_cols:
        if col in encoders:
            # Reuse existing encoder, handle unseen labels
            le = encoders[col]
            df[col] = df[col].map(lambda x, _le=le: _le.transform([x])[0]
                                  if x in _le.classes_ else -1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # --- Interaction feature construction ---
    # OverTime × JobSatisfaction (high workload + low satisfaction synergy)
    if "OverTime" in df.columns and "JobSatisfaction" in df.columns:
        df["OT_x_JobSat"] = df["OverTime"] * df["JobSatisfaction"]
    # MonthlyIncome / (YearsAtCompany + 1) (salary growth per tenure year)
    if "MonthlyIncome" in df.columns and "YearsAtCompany" in df.columns:
        df["Income_per_Year"] = df["MonthlyIncome"] / (df["YearsAtCompany"] + 1)
    # YearsSinceLastPromotion / (YearsAtCompany + 1) (career stagnation)
    if "YearsAtCompany" in df.columns and "YearsSinceLastPromotion" in df.columns:
        df["Promo_Gap_Ratio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    feature_names = df.columns.tolist()

    # --- Z-score standardization ---
    X = df.values.astype(np.float64)
    if fit_scaler is not None:
        scaler = fit_scaler
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_names, encoders, scaler


def split_data(X, y, test_size=0.2):
    """Stratified train/test split."""
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
    print(f"Feature matrix: {X.shape}, target distribution: {np.bincount(y)}")
    print(f"Features: {len(feat_names)}")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
