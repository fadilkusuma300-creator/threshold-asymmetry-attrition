"""
模型定义模块：基线方法与异构集成。

包含：
  - G-Mean 计算、评价指标汇总
  - 阈值优化（自适应搜索最优分类阈值）
  - 基线模型：LR、SVM、RF、XGBoost、LightGBM
  - 异构集成：RF + XGBoost + LightGBM + SVM（等权软投票）
  - SMOTE / ADASYN 增强基线
  - CatBoost + SMOTE、AdaBoost + ADASYN、GradientBoosting + RFE
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
# 评价指标
# ============================================================
def g_mean_score(y_true, y_pred):
    """G-Mean：灵敏度与特异度的几何平均，衡量两类分类均衡性。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


def evaluate(y_true, y_pred, y_prob):
    """返回完整评价指标字典：AUC-ROC、F1、精确率、召回率、G-Mean、Brier。"""
    return {
        "AUC-ROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "G-Mean": g_mean_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
    }


# ============================================================
# 阈值优化
# ============================================================
def predict_with_threshold(y_prob, threshold=0.5):
    """按指定阈值将概率转换为二分类预测。"""
    return (np.asarray(y_prob) >= threshold).astype(int)


def find_optimal_threshold(y_true, y_prob, objective="balanced"):
    """
    在 [0.05, 0.95] 范围内搜索最优分类阈值。

    Parameters
    ----------
    objective : str
        'f1'      — 最大化 F1
        'gmean'   — 最大化 G-Mean
        'balanced' — 加权组合（默认）
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
            # 默认：F1 + G-Mean + AUC 加权组合
            score = 0.45 * metrics["F1"] + 0.35 * metrics["G-Mean"] + 0.20 * metrics["AUC-ROC"]

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
            best_metrics = metrics

    return best_threshold, best_score, best_metrics


# ============================================================
# 基线模型
# ============================================================
def get_baseline_models():
    """返回基线分类器字典（LR、SVM、RF、XGBoost、LightGBM）。"""
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
            scale_pos_weight=5,  # 近似不平衡比率
            n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            is_unbalance=True, n_jobs=-1, verbose=-1,
        ),
    }


# ============================================================
# 异构集成（VotingClassifier 软投票）
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
    构建异构集成：RF + XGBoost + LightGBM (+ 可选 SVM)，等权软投票。

    Parameters
    ----------
    cost_sensitive : bool
        是否启用代价敏感设置（class_weight='balanced'）。
    include_svm : bool
        是否加入 SVM（RBF 核），与树模型决策边界互补。
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
    构建 Stacking 集成（LogisticRegression 作为元学习器）。
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
# SMOTE 增强基线
# ============================================================
def train_global_smote_model(X_train, y_train, model_name="RF"):
    """全局 SMOTE 过采样 + 单模型训练。"""
    k = min(5, np.bincount(y_train).min() - 1)
    k = max(1, k)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = get_baseline_models()
    model = models[model_name]
    model.fit(X_res, y_res)
    return model


# ============================================================
# CatBoost + SMOTE（Raza et al., 2022）
# ============================================================
def train_catboost_smote(X_train, y_train):
    """SMOTE 过采样 + CatBoost 训练。"""
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
    """ADASYN 自适应过采样 + AdaBoost 训练。"""
    n_minority = np.bincount(y_train).min()
    k = min(5, n_minority - 1)
    k = max(1, k)
    try:
        resampler = ADASYN(random_state=RANDOM_STATE, n_neighbors=k)
        X_res, y_res = resampler.fit_resample(X_train, y_train)
    except ValueError:
        # ADASYN 失败时回退到 SMOTE
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
# GradientBoosting + RFE 特征选择（Fang & Zhang, 2024）
# ============================================================
def train_gb_rfe(X_train, y_train, n_features_ratio=0.8):
    """递归特征消除（RFE）+ GradientBoosting 训练。"""
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
# SMOTE + Stacking 集成
# ============================================================
def train_global_smote_stacking(X_train, y_train):
    """全局 SMOTE 过采样 + Stacking 集成训练。"""
    k = min(5, np.bincount(y_train).min() - 1)
    k = max(1, k)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    stacking = build_stacking_ensemble_full(cost_sensitive=False)
    stacking.fit(X_res, y_res)
    return stacking
