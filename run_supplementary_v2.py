#!/usr/bin/env python3
"""
补充基线实验 v2：复现近年文献中的方法。

  - HistGradientBoosting + SMOTE（Al-Ali et al., Scientific Reports, 2026）
  - XGBoost + RFE（Fang & Zhang, IEEE Access, 2024）

使用与 run_final.py 完全相同的数据、预处理、交叉验证折、校准和阈值优化流程。

用法：
  python run_supplementary_v2.py --dataset ibm
  python run_supplementary_v2.py --dataset indian
"""

import sys, os, json, time, warnings, traceback
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RS = 42
N_SPLITS = 5
N_REPEATS = 5
ALPHA = 0.55
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


def make_histgb_smote(rs=RS):
    """HistGradientBoosting + SMOTE pipeline (Al-Ali et al. 2026)."""
    return ImbPipeline([
        ('smote', SMOTE(random_state=rs)),
        ('histgb', HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6,
            random_state=rs, verbose=0)),
    ])


def make_xgb_rfe(rs=RS, n_features_ratio=0.8):
    """XGBoost + RFE feature selection (Fang & Zhang 2024)."""
    # We'll handle RFE manually since it needs fit_transform
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=rs, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1, verbosity=0,
    )


# ============================================================
# Helpers (IDENTICAL to run_final.py)
# ============================================================
def gmean_score(y, yp):
    cm = confusion_matrix(y, yp)
    if cm.size != 4: return 0.0
    tn, fp, fn, tp = cm.ravel()
    s = tp/(tp+fn) if tp+fn else 0
    c = tn/(tn+fp) if tn+fp else 0
    return np.sqrt(s*c)


def evaluate(y, yp, yprob):
    return {
        'AUC': roc_auc_score(y, yprob),
        'F1': f1_score(y, yp, zero_division=0),
        'Precision': precision_score(y, yp, zero_division=0),
        'Recall': recall_score(y, yp, zero_division=0),
        'G-Mean': gmean_score(y, yp),
        'Brier': brier_score_loss(y, yprob),
    }


def find_tau(y, prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_tau, best_s = 0.50, -1.0
    for t in grid:
        yp = (prob >= t).astype(int)
        try:
            f = f1_score(y, yp, zero_division=0)
            g = gmean_score(y, yp)
            s = ALPHA * f + (1-ALPHA) * g
            if s > best_s:
                best_s = s
                best_tau = float(t)
        except:
            pass
    return best_tau


def get_oof_calibrated(model, X, y, cv):
    prob_oof = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    if prob_oof.ndim > 1:
        prob_oof = prob_oof[:, 1]
    try:
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(prob_oof, y)
        return iso.transform(prob_oof), iso
    except:
        return prob_oof, None


def apply_calibration(prob, calibrator):
    if calibrator is not None:
        return calibrator.transform(prob)
    return prob


def pos_prob(proba):
    if proba.ndim > 1:
        return proba[:, 1]
    return proba


# ============================================================
# Data loading (IDENTICAL to run_final.py)
# ============================================================
def load_and_preprocess(name):
    if name == 'IBM':
        fp = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        target_col = 'Attrition'
        drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    elif name == 'Indian':
        fp = DATA_DIR / "hr_attrition_indian.csv"
        target_col = 'LeftCompany'
        drop_cols = ['EmployeeID', 'FullName', 'DateOfJoining', 'City']
    else:
        return None

    if not fp.exists():
        return None

    df = pd.read_csv(fp)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    y = df[target_col].values.astype(int)
    X = df.drop(columns=[target_col])
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = LabelEncoder().fit_transform(X[c])
    if 'OverTime' in X.columns and 'JobSatisfaction' in X.columns:
        X['OT_x_JobSat'] = X['OverTime'] * X['JobSatisfaction']
    if 'MonthlyIncome' in X.columns and 'YearsAtCompany' in X.columns:
        X['Income_per_Year'] = X['MonthlyIncome'] / (X['YearsAtCompany'] + 1)
    if 'YearsSinceLastPromotion' in X.columns and 'YearsAtCompany' in X.columns:
        X['Promo_Gap_Ratio'] = X['YearsSinceLastPromotion'] / (X['YearsAtCompany'] + 1)
    X_scaled = StandardScaler().fit_transform(X.values.astype(np.float64))
    ir = (y==0).sum() / max((y==1).sum(), 1)
    print(f"[DATA] {name}: ({len(y)}, {X.shape[1]}), IR={ir:.2f}:1", flush=True)
    return X_scaled, y, X.shape[1], ir


def add_cluster_features(X_train, X_test):
    best_k, best_sil = 2, -1
    for k in range(2, 7):
        try:
            km = KMeans(n_clusters=k, random_state=RS, n_init=10)
            s = silhouette_score(X_train, km.fit_predict(X_train))
            if s > best_sil:
                best_sil = s
                best_k = k
        except:
            pass
    km = KMeans(n_clusters=best_k, random_state=RS, n_init=10).fit(X_train)
    centers = km.cluster_centers_
    def dists(X):
        return np.column_stack([np.linalg.norm(X - c, axis=1) for c in centers])
    return np.hstack([X_train, dists(X_train)]), np.hstack([X_test, dists(X_test)]), best_k


# ============================================================
# Main experiment
# ============================================================
def run_supplementary(X, y, name, ir):
    t0 = time.time()
    n = len(y)
    print(f"\n{'='*70}", flush=True)
    print(f"  SUPPLEMENTARY: {name} — HistGB+SMOTE & XGB+RFE", flush=True)
    print(f"  {N_REPEATS} repeats × {N_SPLITS} folds = {N_REPEATS*N_SPLITS}", flush=True)
    print(f"{'='*70}", flush=True)

    R = {'c1': [], 'c2': [], 'c3': [], 'thresholds': []}
    total = N_REPEATS * N_SPLITS
    count = 0

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)
        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            count += 1
            if count % 5 == 1 or count == total:
                print(f"\n  [{count}/{total}] Rep {rep+1}, Fold {fold+1}", flush=True)

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_aug, X_te_aug, K = add_cluster_features(X_tr, X_te)

            methods = {}

            # ---- 1. HistGradientBoosting + SMOTE ----
            try:
                hg = make_histgb_smote()
                oof_hg, iso_hg = get_oof_calibrated(hg, X_tr_aug, y_tr, inner_cv)
                tau_hg = find_tau(y_tr, oof_hg)
                hg.fit(X_tr_aug, y_tr)
                test_prob_hg = apply_calibration(pos_prob(hg.predict_proba(X_te_aug)), iso_hg)
                methods['HistGB+SMOTE'] = {'prob': test_prob_hg, 'tau': tau_hg}
                print(f"    HistGB+SMOTE OK: tau*={tau_hg:.3f}", flush=True)
            except Exception as e:
                print(f"    HistGB+SMOTE FAILED: {e}", flush=True)
                traceback.print_exc()

            # ---- 2. XGBoost + RFE ----
            try:
                n_features = max(5, int(X_tr_aug.shape[1] * 0.8))
                selector = XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=RS, use_label_encoder=False,
                    eval_metric='logloss', n_jobs=-1, verbosity=0,
                )
                rfe = RFE(estimator=selector, n_features_to_select=n_features, step=1)
                rfe.fit(X_tr_aug, y_tr)
                X_tr_sel = rfe.transform(X_tr_aug)
                X_te_sel = rfe.transform(X_te_aug)

                xgb_rfe = make_xgb_rfe()
                oof_xr, iso_xr = get_oof_calibrated(xgb_rfe, X_tr_sel, y_tr, inner_cv)
                tau_xr = find_tau(y_tr, oof_xr)
                xgb_rfe.fit(X_tr_sel, y_tr)
                test_prob_xr = apply_calibration(pos_prob(xgb_rfe.predict_proba(X_te_sel)), iso_xr)
                methods['XGBoost+RFE'] = {'prob': test_prob_xr, 'tau': tau_xr}
                print(f"    XGBoost+RFE OK: tau*={tau_xr:.3f} ({n_features} features)", flush=True)
            except Exception as e:
                print(f"    XGBoost+RFE FAILED: {e}", flush=True)
                traceback.print_exc()

            # Store
            R['thresholds'].append({m: d['tau'] for m, d in methods.items()})

            fold_c1, fold_c2, fold_c3 = {}, {}, {}
            for mname, mdata in methods.items():
                prob = mdata['prob']
                tau_opt = mdata['tau']
                fold_c1[mname] = evaluate(y_te, (prob >= 0.50).astype(int), prob)
                fold_c2[mname] = evaluate(y_te, (prob >= tau_opt).astype(int), prob)
                fold_c3[mname] = evaluate(y_te, (prob >= 0.50).astype(int), prob)

            R['c1'].append(fold_c1)
            R['c2'].append(fold_c2)
            R['c3'].append(fold_c3)

    total_time = time.time() - t0
    print(f"\n[TIME] {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)

    # Aggregate
    def agg(ck):
        all_m = set()
        for fd in R[ck]:
            all_m.update(fd.keys())
        result = {}
        for mn in sorted(all_m):
            metrics = {'AUC':[], 'F1':[], 'Precision':[], 'Recall':[], 'G-Mean':[], 'Brier':[]}
            for fd in R[ck]:
                if mn in fd:
                    for m in metrics:
                        metrics[m].append(fd[mn][m])
            result[mn] = {}
            for m in metrics:
                vals = np.array(metrics[m])
                result[mn][m] = {'mean': round(float(np.mean(vals)), 4),
                                  'std': round(float(np.std(vals, ddof=1)), 4)}
        return result

    results = {
        'dataset': name,
        'n_samples': n,
        'n_features': X.shape[1],
        'n_repeats': N_REPEATS,
        'n_folds': N_SPLITS,
        'imbalance_ratio': round(ir, 2),
        'optimal_thresholds': {},
        'three_condition': {
            'c1_equal_tau050': agg('c1'),
            'c2_fair_tau': agg('c2'),
            'c3_asymmetric': agg('c3'),
        },
    }

    all_m = set()
    for d in R['thresholds']:
        all_m.update(d.keys())
    for m in sorted(all_m):
        vals = [d.get(m) for d in R['thresholds'] if m in d]
        if vals:
            results['optimal_thresholds'][m] = {
                'mean': round(float(np.mean(vals)), 3),
                'std': round(float(np.std(vals, ddof=1)), 3),
            }

    out = RESULTS_DIR / f'supplementary_v2_results_{name.lower()}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out}", flush=True)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  {name} — SUPPLEMENTARY V2 RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    for m, t in results['optimal_thresholds'].items():
        print(f"  {m}: tau* = {t['mean']:.3f} ± {t['std']:.3f}", flush=True)

    for label, key in [('C1 (tau=0.50)', 'c1_equal_tau050'),
                        ('C2 (fair tau*)', 'c2_fair_tau'),
                        ('C3 (asymmetric)', 'c3_asymmetric')]:
        print(f"\n  --- {label} ---", flush=True)
        cd = results['three_condition'][key]
        for m in sorted(cd.keys()):
            v = cd[m]
            print(f"    {m:20s}  AUC={v['AUC']['mean']:.4f}  F1={v['F1']['mean']:.4f}±{v['F1']['std']:.3f}  "
                  f"P={v['Precision']['mean']:.4f}  R={v['Recall']['mean']:.4f}  "
                  f"GM={v['G-Mean']['mean']:.4f}  Brier={v['Brier']['mean']:.4f}",
                  flush=True)

    return results


if __name__ == "__main__":
    print("="*70, flush=True)
    print("  Supplementary Baselines v2", flush=True)
    print("  HistGB+SMOTE & XGB+RFE", flush=True)
    print("="*70, flush=True)

    result1 = load_and_preprocess('IBM')
    if result1:
        X1, y1, nfeat1, ir1 = result1
        run_supplementary(X1, y1, 'IBM', ir1)

    result2 = load_and_preprocess('Indian')
    if result2:
        X2, y2, nfeat2, ir2 = result2
        run_supplementary(X2, y2, 'Indian', ir2)

    print(f"\n{'='*70}", flush=True)
    print("  ALL COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
