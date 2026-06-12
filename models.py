"""
Model definitions: baselines and heterogeneous ensemble.

Contains:
  - G-Mean computation, evaluation metrics
  - Threshold optimization (adaptive search for optimal classification threshold)
  - Baseline models: LR, SVM, RF, XGBoost, LightGBM
  - Heterogeneous ensemble: RF + XGBoost + LightGBM + SVM (equal-weight soft voting)
  - SMOTE / ADASYN augmented baselines
  - CatBoost + SMOTE, AdaBoost + ADASYN, GradientBoosting + RFE
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, VotingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
from config import RANDOM_STATE


# ============================================================
# Evaluation metrics
# ============================================================
def g_mean_score(y_true, y_pred):
    """G-Mean: geometric mean of sensitivity and specificity."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


def evaluate(y_true, y_pred, y_prob):
    """Return full evaluation metrics: AUC-ROC, F1, Precision, Recall, G-Mean, Brier."""
    return {
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "G-Mean": g_mean_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
    }


# ============================================================
# Threshold optimization
# ============================================================
def predict_with_threshold(y_prob, threshold=0.5):
    """Convert probabilities to binary predictions using a given threshold."""
    return (np.asarray(y_prob) >= threshold).astype(int)


def find_optimal_threshold(y_true, y_prob, objective="balanced"):
    """
    Search for the optimal classification threshold in [0.05, 0.95].

    Parameters
    ----------
    objective : str
        'f1'       — maximize F1
        'gmean'    — maximize G-Mean
        'balanced' — weighted combination (default)
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for threshold in thresholds:
        y_pred = predict_with_threshold(y_prob, threshold)
        metrics = evaluate(y_true, y_pred, y_prob)
        if objective == "f1":
            score = metrics["F1"]
        elif objective == "gmean":
            score = metrics["G-Mean"]
        else:
            # Default: weighted combination of F1 + G-Mean + AUC
            score = 0.45 * metrics["F1"] + 0.35 * metrics["G-Mean"] + 0.20 * metrics["AUC-ROC"]

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


# ============================================================
# Baseline models
# ============================================================
def get_baseline_models():
    """Return baseline classifier dict (LR, SVM, RF, XGBoost, LightGBM)."""
    return {
        "LR": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=5,  # approximate imbalance ratio
            n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            is_unbalance=True, n_jobs=-1, verbose=-1,
        ),
    }


# ============================================================
# Heterogeneous ensemble (VotingClassifier, soft voting)
# ============================================================
def build_stacking_ensemble(
    cost_sensitive: bool = False,
    weights=None,
    scale_pos_weight=None,
    xgb_max_depth=3,
    lgbm_num_leaves=31,
    include_svm=False,
    svm_C=1.0,
):
    """
    Build heterogeneous ensemble: RF + XGBoost + LightGBM (+ optional SVM),
    equal-weight soft voting.

    Parameters
    ----------
    cost_sensitive : bool
        Enable cost-sensitive settings (class_weight='balanced').
    include_svm : bool
        Include SVM (RBF kernel) for complementary decision boundaries.
    """
    cw = "balanced" if cost_sensitive else None
    spw = scale_pos_weight if scale_pos_weight is not None else (5 if cost_sensitive else 1)

    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight=cw, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=spw, max_depth=xgb_max_depth,
            n_jobs=-1, verbosity=0,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            scale_pos_weight=spw, num_leaves=lgbm_num_leaves,
            n_jobs=-1, verbose=-1,
        )),
    ]

    if include_svm:
        base_estimators.append(("svm", SVC(
            kernel="rbf", C=svm_C, probability=True,
            random_state=RANDOM_STATE, class_weight=cw,
        )))

    ensemble = VotingClassifier(
        estimators=base_estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )
    return ensemble


def build_stacking_ensemble_full(cost_sensitive: bool = False):
    """
    Build Stacking ensemble (LogisticRegression as meta-learner).
    """
    cw = "balanced" if cost_sensitive else None
    spw = 5 if cost_sensitive else 1

    base_estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight=cw, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=spw, n_jobs=-1, verbosity=0,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            is_unbalance=cost_sensitive, n_jobs=-1, verbose=-1,
        )),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return stacking


# ============================================================
# SMOTE-augmented baselines
# ============================================================
def train_global_smote_model(X_train, y_train, model_name="RF"):
    """Global SMOTE oversampling + single model training."""
    k = min(5, np.bincount(y_train).min() - 1)
    k = max(1, k)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = get_baseline_models()
    model = models[model_name]
    model.fit(X_res, y_res)
    return model


# ============================================================
# CatBoost + SMOTE (Raza et al., 2022)
# ============================================================
def train_catboost_smote(X_train, y_train):
    """SMOTE oversampling + CatBoost training."""
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed")
    k = min(5, np.bincount(y_train).min() - 1)
    k = max(1, k)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=6,
        random_seed=RANDOM_STATE, verbose=0,
        auto_class_weights="Balanced",
    )
    model.fit(X_res, y_res)
    return model


# ============================================================
# AdaBoost + ADASYN
# ============================================================
def train_adaboost_adasyn(X_train, y_train):
    """ADASYN adaptive oversampling + AdaBoost training."""
    n_minority = np.bincount(y_train).min()
    k = min(5, n_minority - 1)
    k = max(1, k)
    try:
        resampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=k)
        X_res, y_res = resampler.fit_resample(X_train, y_train)
    except ValueError:
        # Fall back to SMOTE if ADASYN fails
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)

    base_dt = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    model = AdaBoostClassifier(
        estimator=base_dt,
        n_estimators=200, learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_res, y_res)
    return model


# ============================================================
# GradientBoosting + RFE (Fang & Zhang, 2024)
# ============================================================
def train_gb_rfe(X_train, y_train, n_features_ratio=0.8):
    """Recursive Feature Elimination (RFE) + GradientBoosting training."""
    n_features = max(5, int(X_train.shape[1] * n_features_ratio))

    selector_model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=RANDOM_STATE,
    )
    rfe = RFE(estimator=selector_model, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    X_sel = rfe.transform(X_train)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_STATE, min_samples_split=10,
    )
    model.fit(X_sel, y_train)
    return model, rfe


# ============================================================
# SMOTE + Stacking ensemble
# ============================================================
def train_global_smote_stacking(X_train, y_train):
    """Global SMOTE oversampling + Stacking ensemble training."""
    k = min(5, np.bincount(y_train).min() - 1)
    k = max(1, k)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    stacking = build_stacking_ensemble_full(cost_sensitive=False)
    stacking.fit(X_res, y_res)
    return stacking
