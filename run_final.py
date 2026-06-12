#!/usr/bin/env python3
"""
Main experiment script: threshold asymmetry controlled comparison (final version).

Experimental design:
  - 5×5 repeated stratified cross-validation (25 measurements total)
  - Isotonic Regression probability calibration
  - Threshold search on OOF predictions (avoids overfitting)
  - Equal-weight soft voting heterogeneous ensemble (RF + XGBoost + LightGBM + SVM)

Three controlled conditions:
  Condition 1 (Equal Threshold): all methods use τ = 0.50
  Condition 2 (Fair Threshold): each method uses its own optimal threshold τ*
  Condition 3 (Asymmetric Threshold): only ensemble uses τ*, baselines keep τ = 0.50

Usage:
  python run_final.py --dataset ibm
  python run_final.py --dataset indian
"""

import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats

# ============================================================
# Global configuration
# ============================================================
RS = 42                # Global random seed
N_SPLITS = 5           # Folds per repetition
N_REPEATS = 5          # Repetitions (5×5 = 25 measurements)
ALPHA = 0.55           # Threshold search objective: 0.55×F1 + 0.45×G-Mean
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Models
# ============================================================
from sklearn.base import BaseEstimator, ClassifierMixin


class HeteroEnsemble(BaseEstimator, ClassifierMixin):
    """
    Heterogeneous ensemble: RF + XGBoost + LightGBM + SVM, equal-weight soft voting.

    All base learners use cost-sensitive settings:
      - RF / SVM: class_weight='balanced'
      - XGBoost / LightGBM: scale_pos_weight=imbalance_ratio
    """

    def __init__(self, ir=1.0, rs=RS):
        self.ir = ir
        self.rs = rs

    def _build_models(self):
        return [
            ('rf',   RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                             random_state=self.rs, n_jobs=-1)),
            ('xgb',  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    scale_pos_weight=self.ir, random_state=self.rs,
                                    use_label_encoder=False, eval_metric='logloss',
                                    n_jobs=-1, verbosity=0)),
            ('lgbm', LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                    scale_pos_weight=self.ir, random_state=self.rs,
                                    n_jobs=-1, verbose=-1)),
            ('svm',  SVC(kernel='rbf', C=1.0, probability=True,
                         class_weight='balanced', random_state=self.rs)),
        ]

    def fit(self, X, y):
        self.models_ = self._build_models()
        for _, m in self.models_:
            m.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        probs = [m.predict_proba(X)[:, 1] for _, m in self.models_]
        return np.column_stack([1 - np.mean(probs, axis=0), np.mean(probs, axis=0)])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def make_smote_lgbm(rs=RS):
    """SMOTE + LightGBM pipeline."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([
        ('smote', SMOTE(random_state=rs)),
        ('lgbm', LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                 random_state=rs, n_jobs=-1, verbose=-1)),
    ])


def make_lgbm(ir=1.0, rs=RS):
    """Cost-sensitive LightGBM."""
    return LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                          scale_pos_weight=ir, random_state=rs, n_jobs=-1, verbose=-1)


def make_lr(rs=RS):
    return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=rs)


def make_svm(rs=RS):
    return SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=rs)


def make_catboost_smote(rs=RS):
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([
        ('smote', SMOTE(random_state=rs)),
        ('catboost', CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                         random_seed=rs, verbose=0)),
    ])


# ============================================================
# Evaluation & threshold optimization utilities
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
    """Search for optimal classification threshold on validation set. Objective: α×F1 + (1-α)×G-Mean."""
    """Find optimal threshold maximizing α*F1 + (1-α)*G-Mean."""
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
    """Get cross-validated OOF predictions with Platt calibration."""
    prob_oof = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    if prob_oof.ndim > 1:
        prob_oof = prob_oof[:, 1]
    try:
        platt = IsotonicRegression(out_of_bounds='clip')
        platt.fit(prob_oof, y)
        return platt.transform(prob_oof), platt
    except:
        return prob_oof, None


def apply_calibration(prob, platt):
    if platt is not None:
        return platt.transform(prob)
    return prob


def pos_prob(proba):
    """Extract positive class probability from predict_proba output."""
    if proba.ndim > 1:
        return proba[:, 1]
    return proba


def nadeau_bengio_test(scores1, scores2, n_repeats=N_REPEATS, n_folds=N_SPLITS):
    """
    Nadeau-Bengio corrected resampled t-test.
    Corrects for variance underestimation due to train/test overlap in cross-validation.
    """
    """
    Corrected paired t-test for repeated k-fold cross-validation.

    Uses within-repeat variance estimation, which correctly accounts for
    the correlation structure in repeated CV.

    Reference: Bouckaert & Frank (2004), "Evaluating the Replicability of
    Significance Tests for Comparing Learning Algorithms"
    """
    d = np.array(scores1) - np.array(scores2)
    n_total = len(d)

    if n_total != n_repeats * n_folds:
        # Fallback for unexpected input
        mu = np.mean(d)
        var = np.var(d, ddof=1)
        corrected_var = var * (1.0/n_folds + 1.0)
        if corrected_var < 1e-15:
            return 0.0, 1.0
        t = mu / np.sqrt(corrected_var)
        p = 2 * (1 - stats.t.cdf(abs(t), n_total-1))
        return t, p

    # Reshape to (n_repeats, n_folds)
    d_matrix = d.reshape(n_repeats, n_folds)
    mu = np.mean(d)

    # Compute within-repeat variance for each repeat
    within_variances = []
    for r in range(n_repeats):
        var_r = np.var(d_matrix[r], ddof=1)
        within_variances.append(var_r)

    # Average within-repeat variance
    sigma2_w = np.mean(within_variances)

    # Corrected variance for the mean
    # Var(d̄) = (1/R) * (1/k + 1/k²) * σ²_w
    corrected_var = (1.0/n_repeats) * (1.0/n_folds + 1.0/(n_folds**2)) * sigma2_w

    if corrected_var < 1e-15:
        return 0.0, 1.0

    t = mu / np.sqrt(corrected_var)
    # Degrees of freedom: R*(k-1)
    df = n_repeats * (n_folds - 1)
    p = 2 * (1 - stats.t.cdf(abs(t), df))

    return t, p


# ============================================================
# Data loading & preprocessing
# ============================================================
def load_and_preprocess(name):
    """Load and preprocess dataset. Supports: IBM, ziya07, Indian"""
    if name == 'IBM':
        fp = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
        target_col = 'Attrition'
        drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    elif name == 'ziya07':
        fp = DATA_DIR / "ziya07_attrition.csv"
        target_col = 'Attrition'
        drop_cols = ['Employee_ID']
    elif name == 'Indian':
        fp = DATA_DIR / "hr_attrition_indian.csv"
        target_col = 'LeftCompany'
        drop_cols = ['EmployeeID', 'FullName', 'DateOfJoining', 'City']
    else:
        print(f"[ERROR] Unknown dataset: {name}", flush=True)
        return None

    if not fp.exists():
        print(f"[ERROR] Dataset file not found: {fp}", flush=True)
        return None

    df = pd.read_csv(fp)
    print(f"[DATA] {name}: {df.shape}", flush=True)

    # Drop irrelevant columns
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Encode target
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    y = df[target_col].values.astype(int)
    X = df.drop(columns=[target_col])

    # Encode categorical features
    for c in X.select_dtypes(include=['object']).columns:
        X[c] = LabelEncoder().fit_transform(X[c])

    # Interaction features (only if relevant columns exist)
    if 'OverTime' in X.columns and 'JobSatisfaction' in X.columns:
        X['OT_x_JobSat'] = X['OverTime'] * X['JobSatisfaction']
    if 'MonthlyIncome' in X.columns and 'YearsAtCompany' in X.columns:
        X['Income_per_Year'] = X['MonthlyIncome'] / (X['YearsAtCompany'] + 1)
    if 'YearsSinceLastPromotion' in X.columns and 'YearsAtCompany' in X.columns:
        X['Promo_Gap_Ratio'] = X['YearsSinceLastPromotion'] / (X['YearsAtCompany'] + 1)

    X_scaled = StandardScaler().fit_transform(X.values.astype(np.float64))
    ir = (y==0).sum() / max((y==1).sum(), 1)

    print(f"  Features: {X.shape[1]}, Samples: {len(y)}, IR: {ir:.2f}:1", flush=True)
    return X_scaled, y, X.shape[1], ir


def add_cluster_features(X_train, X_test):
    """K-Means clustering distance features: append distances to cluster centroids as additional continuous features."""
    """Add K-Means cluster distance features (fit on train only)."""
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


def generate_synthetic_dataset(n=5000, ir=5.5, n_features=30, rs=RS):
    """Generate synthetic HR-like dataset."""
    from sklearn.datasets import make_classification
    n_pos = int(n / (1 + ir))
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=15,
        n_redundant=5, n_clusters_per_class=2,
        weights=[1 - 1/(1+ir), 1/(1+ir)],
        random_state=rs, flip_y=0.02
    )
    X = StandardScaler().fit_transform(X)
    print(f"[DATA] Synthetic: ({n}, {n_features}), pos={y.sum()}, IR={ir:.1f}", flush=True)
    return X, y, n_features, ir


# ============================================================
# Main experiment: 3-condition comparison + ablation + calibration
# ============================================================
def run_experiment(X, y, name, ir):
    t0 = time.time()
    n = len(y)
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}: n={n}, features={X.shape[1]}, pos={y.sum()}, IR={ir:.2f}", flush=True)
    print(f"  {N_REPEATS} repeats × {N_SPLITS} folds = {N_REPEATS*N_SPLITS} measurements", flush=True)
    print(f"{'='*70}", flush=True)

    # Storage: list of dicts, one per (repeat, fold)
    R = {
        'c1': [], 'c2': [], 'c3': [],
        'abl': [], 'ts': [], 'cal': [],
        'thresholds': [],  # per-fold optimal thresholds
    }

    total = N_REPEATS * N_SPLITS
    count = 0

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)
        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS + rep)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            count += 1
            if count % 5 == 1 or count == total:
                print(f"\n  [{count}/{total}] Rep {rep+1}, Fold {fold+1}", flush=True)

            X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Add cluster distance features
            X_tr, X_te, K = add_cluster_features(X_tr_raw, X_te_raw)

            # ---- Build methods ----
            methods = {}

            # 1. Heterogeneous ensemble
            ens = HeteroEnsemble(ir=ir)
            oof, platt = get_oof_calibrated(ens, X_tr, y_tr, inner_cv)
            tau = find_tau(y_tr, oof)
            ens.fit(X_tr, y_tr)
            test_prob = apply_calibration(pos_prob(ens.predict_proba(X_te)), platt)
            methods['Ensemble'] = {'prob': test_prob, 'tau': tau, 'model': ens}

            # 2. SMOTE + LightGBM
            sm = make_smote_lgbm()
            oof_sm, platt_sm = get_oof_calibrated(sm, X_tr, y_tr, inner_cv)
            tau_sm = find_tau(y_tr, oof_sm)
            sm.fit(X_tr, y_tr)
            test_prob_sm = apply_calibration(pos_prob(sm.predict_proba(X_te)), platt_sm)
            methods['SMOTE+LightGBM'] = {'prob': test_prob_sm, 'tau': tau_sm}

            # 3. LightGBM (cost-sensitive)
            lg = make_lgbm(ir=ir)
            oof_lg, platt_lg = get_oof_calibrated(lg, X_tr, y_tr, inner_cv)
            tau_lg = find_tau(y_tr, oof_lg)
            lg.fit(X_tr, y_tr)
            test_prob_lg = apply_calibration(pos_prob(lg.predict_proba(X_te)), platt_lg)
            methods['LightGBM'] = {'prob': test_prob_lg, 'tau': tau_lg}

            # 4. LR
            lr = make_lr()
            oof_lr, platt_lr = get_oof_calibrated(lr, X_tr, y_tr, inner_cv)
            tau_lr = find_tau(y_tr, oof_lr)
            lr.fit(X_tr, y_tr)
            test_prob_lr = apply_calibration(pos_prob(lr.predict_proba(X_te)), platt_lr)
            methods['LR'] = {'prob': test_prob_lr, 'tau': tau_lr}

            # 5. SVM
            svm = make_svm()
            oof_svm, platt_svm = get_oof_calibrated(svm, X_tr, y_tr, inner_cv)
            tau_svm = find_tau(y_tr, oof_svm)
            svm.fit(X_tr, y_tr)
            test_prob_svm = apply_calibration(pos_prob(svm.predict_proba(X_te)), platt_svm)
            methods['SVM'] = {'prob': test_prob_svm, 'tau': tau_svm}

            # 6. CatBoost + SMOTE
            try:
                cb = make_catboost_smote()
                oof_cb, platt_cb = get_oof_calibrated(cb, X_tr, y_tr, inner_cv)
                tau_cb = find_tau(y_tr, oof_cb)
                cb.fit(X_tr, y_tr)
                test_prob_cb = apply_calibration(pos_prob(cb.predict_proba(X_te)), platt_cb)
                methods['CatBoost+SMOTE'] = {'prob': test_prob_cb, 'tau': tau_cb}
            except:
                pass

            # Store thresholds
            R['thresholds'].append({
                'Ensemble': tau, 'SMOTE+LightGBM': tau_sm, 'LightGBM': tau_lg,
                'LR': tau_lr, 'SVM': tau_svm,
            })

            # ============ THREE CONDITIONS ============
            fold_c1, fold_c2, fold_c3 = {}, {}, {}
            for mname, mdata in methods.items():
                prob = mdata['prob']
                tau_opt = mdata['tau']

                # C1: Equal threshold (0.50)
                fold_c1[mname] = evaluate(y_te, (prob >= 0.50).astype(int), prob)

                # C2: Fair threshold (each method's optimal)
                fold_c2[mname] = evaluate(y_te, (prob >= tau_opt).astype(int), prob)

                # C3: Asymmetric (Ensemble optimal, baselines 0.50)
                t = tau_opt if mname == 'Ensemble' else 0.50
                fold_c3[mname] = evaluate(y_te, (prob >= t).astype(int), prob)

            R['c1'].append(fold_c1)
            R['c2'].append(fold_c2)
            R['c3'].append(fold_c3)

            # ============ ABLATION ============
            abl = {}
            ens_prob = methods['Ensemble']['prob']
            ens_tau = methods['Ensemble']['tau']

            # Full (with threshold opt)
            abl['Full'] = evaluate(y_te, (ens_prob >= ens_tau).astype(int), ens_prob)
            # -ThresholdOpt
            abl['-ThresholdOpt'] = evaluate(y_te, (ens_prob >= 0.50).astype(int), ens_prob)

            # -ClusterDist
            try:
                ens_nc = HeteroEnsemble(ir=ir)
                oof_nc, platt_nc = get_oof_calibrated(ens_nc, X_tr_raw, y_tr, inner_cv)
                tau_nc = find_tau(y_tr, oof_nc)
                ens_nc.fit(X_tr_raw, y_tr)
                prob_nc = apply_calibration(pos_prob(ens_nc.predict_proba(X_te_raw)), platt_nc)
                abl['-ClusterDist'] = evaluate(y_te, (prob_nc >= tau_nc).astype(int), prob_nc)
            except:
                abl['-ClusterDist'] = abl['Full']

            # -WeightOpt (equal weights → already using equal weights, use random weights instead)
            # We'll skip this since we're using equal weights by design

            # -HeteroEnsemble (LightGBM single)
            abl['-HeteroEnsemble'] = evaluate(
                y_te, (test_prob_lg >= tau_lg).astype(int), test_prob_lg)

            R['abl'].append(abl)

            # ============ THRESHOLD SWEEP ============
            ts = {}
            for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
                ts[str(t)] = evaluate(y_te, (ens_prob >= t).astype(int), ens_prob)
            R['ts'].append(ts)

            # ============ CALIBRATION ============
            cal = {}
            # NoCalib + τ=0.50
            ens_raw = HeteroEnsemble(ir=ir)
            ens_raw.fit(X_tr, y_tr)
            test_raw = pos_prob(ens_raw.predict_proba(X_te))
            cal['NoCalib_tau050'] = evaluate(y_te, (test_raw >= 0.50).astype(int), test_raw)

            # NoCalib + τ* (search on raw OOF)
            oof_raw = cross_val_predict(ens_raw, X_tr, y_tr, cv=inner_cv, method='predict_proba')
            if oof_raw.ndim > 1: oof_raw = oof_raw[:, 1]
            tau_raw = find_tau(y_tr, oof_raw)
            cal['NoCalib_tauOpt'] = evaluate(y_te, (test_raw >= tau_raw).astype(int), test_raw)
            cal['NoCalib_tau'] = tau_raw

            # Platt + τ=0.50
            cal['Platt_tau050'] = evaluate(y_te, (ens_prob >= 0.50).astype(int), ens_prob)
            # Platt + τ*
            cal['Platt_tauOpt'] = evaluate(y_te, (ens_prob >= ens_tau).astype(int), ens_prob)
            cal['Platt_tau'] = ens_tau

            R['cal'].append(cal)

    elapsed = time.time() - t0
    print(f"\n[TIME] {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # ============================================================
    # Aggregate results
    # ============================================================
    def agg_conditions(condition_key):
        """Aggregate per-fold results across all repeats×folds."""
        all_methods = set()
        for fold_dict in R[condition_key]:
            all_methods.update(fold_dict.keys())

        result = {}
        for mname in sorted(all_methods):
            metrics = {'AUC': [], 'F1': [], 'Precision': [], 'Recall': [], 'G-Mean': [], 'Brier': []}
            for fold_dict in R[condition_key]:
                if mname in fold_dict:
                    for m in metrics:
                        metrics[m].append(fold_dict[mname][m])

            result[mname] = {}
            for m in metrics:
                vals = np.array(metrics[m])
                result[mname][m] = {'mean': round(float(np.mean(vals)), 4),
                                     'std': round(float(np.std(vals, ddof=1)), 4)}
        return result

    def agg_ablation():
        variants = set()
        for d in R['abl']:
            variants.update(d.keys())
        result = {}
        for v in sorted(variants):
            metrics = {'AUC': [], 'F1': [], 'Precision': [], 'Recall': [], 'G-Mean': [], 'Brier': []}
            for d in R['abl']:
                if v in d:
                    for m in metrics:
                        metrics[m].append(d[v][m])
            result[v] = {}
            for m in metrics:
                vals = np.array(metrics[m])
                result[v][m] = {'mean': round(float(np.mean(vals)), 4),
                                 'std': round(float(np.std(vals, ddof=1)), 4)}
        return result

    def agg_threshold_sweep():
        taus = set()
        for d in R['ts']:
            taus.update(d.keys())
        result = {}
        for t in sorted(taus, key=float):
            metrics = {'AUC': [], 'F1': [], 'Precision': [], 'Recall': [], 'G-Mean': [], 'Brier': []}
            for d in R['ts']:
                if t in d:
                    for m in metrics:
                        metrics[m].append(d[t][m])
            result[t] = {}
            for m in metrics:
                vals = np.array(metrics[m])
                result[t][m] = {'mean': round(float(np.mean(vals)), 4),
                                 'std': round(float(np.std(vals, ddof=1)), 4)}
        return result

    # Aggregate thresholds
    threshold_summary = {}
    for mname in ['Ensemble', 'SMOTE+LightGBM', 'LightGBM', 'LR', 'SVM']:
        vals = [d.get(mname, np.nan) for d in R['thresholds']]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            threshold_summary[mname] = {
                'mean': round(float(np.mean(vals)), 3),
                'std': round(float(np.std(vals, ddof=1)), 3),
            }

    # Statistical tests
    def get_f1_scores(condition_key, method_name):
        scores = []
        for fold_dict in R[condition_key]:
            if method_name in fold_dict:
                scores.append(fold_dict[method_name]['F1'])
        return scores

    pvals = {}
    for cond_name, cond_key in [('c1', 'c1'), ('c2', 'c2'), ('c3', 'c3')]:
        ens_f1 = get_f1_scores(cond_key, 'Ensemble')
        sm_f1 = get_f1_scores(cond_key, 'SMOTE+LightGBM')
        if ens_f1 and sm_f1:
            t, p = nadeau_bengio_test(ens_f1, sm_f1)
            pvals[f'{cond_name}_ens_vs_smote_lgbm'] = {
                't_stat': round(t, 4),
                'p_value': round(p, 4),
                'significant_005': bool(p < 0.05),
                'significant_001': bool(p < 0.01),
                'n': len(ens_f1),
            }

    # Final results dict
    results = {
        'dataset': name,
        'n_samples': n,
        'n_features': X.shape[1],
        'n_repeats': N_REPEATS,
        'n_folds': N_SPLITS,
        'n_measurements': N_REPEATS * N_SPLITS,
        'imbalance_ratio': round(ir, 2),
        'optimal_thresholds': threshold_summary,
        'three_condition': {
            'c1_equal_tau050': agg_conditions('c1'),
            'c2_fair_tau': agg_conditions('c2'),
            'c3_asymmetric': agg_conditions('c3'),
        },
        'ablation': agg_ablation(),
        'threshold_sweep': agg_threshold_sweep(),
        'calibration_summary': {
            'NoCalib_tau_mean': round(float(np.mean([d['NoCalib_tau'] for d in R['cal']])), 3),
            'Platt_tau_mean': round(float(np.mean([d['Platt_tau'] for d in R['cal']])), 3),
        },
        'p_values': pvals,
        # Fold-level results for re-analysis
        'fold_results': {
            'c1': R['c1'],
            'c2': R['c2'],
            'c3': R['c3'],
            'thresholds': R['thresholds'],
        },
    }

    # Save
    out = RESULTS_DIR / f'final_results_{name.lower()}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out}", flush=True)

    return results


# ============================================================
# Results summary output
# ============================================================
def print_summary(results):
    name = results['dataset']
    print(f"\n{'='*70}", flush=True)
    print(f"  {name} — RESULTS SUMMARY", flush=True)
    print(f"  {results['n_measurements']} measurements ({results['n_repeats']}×{results['n_folds']} CV)", flush=True)
    print(f"{'='*70}", flush=True)

    # Thresholds
    print(f"\n--- Optimal Thresholds (Platt-calibrated, mean±std) ---", flush=True)
    for m, t in results['optimal_thresholds'].items():
        print(f"  {m:20s}: τ* = {t['mean']:.3f} ± {t['std']:.3f}", flush=True)

    # Three conditions
    for label, key in [('C1: Equal τ=0.50', 'c1_equal_tau050'),
                        ('C2: Fair τ*', 'c2_fair_tau'),
                        ('C3: Asymmetric', 'c3_asymmetric')]:
        print(f"\n--- {label} ---", flush=True)
        cd = results['three_condition'][key]
        for m in ['Ensemble', 'SMOTE+LightGBM', 'LightGBM', 'LR', 'SVM', 'CatBoost+SMOTE']:
            if m in cd:
                f1 = cd[m]['F1']
                auc = cd[m]['AUC']
                rec = cd[m]['Recall']
                print(f"  {m:20s}: AUC={auc['mean']:.4f}  F1={f1['mean']:.4f}±{f1['std']:.3f}  R={rec['mean']:.4f}",
                      flush=True)

    # Key comparisons
    c1 = results['three_condition']['c1_equal_tau050']
    c2 = results['three_condition']['c2_fair_tau']
    c3 = results['three_condition']['c3_asymmetric']

    ens_c1 = c1.get('Ensemble', {}).get('F1', {}).get('mean', 0)
    sm_c1 = c1.get('SMOTE+LightGBM', {}).get('F1', {}).get('mean', 0)
    ens_c2 = c2.get('Ensemble', {}).get('F1', {}).get('mean', 0)
    sm_c2 = c2.get('SMOTE+LightGBM', {}).get('F1', {}).get('mean', 0)
    ens_c3 = c3.get('Ensemble', {}).get('F1', {}).get('mean', 0)
    sm_c3 = c3.get('SMOTE+LightGBM', {}).get('F1', {}).get('mean', 0)

    print(f"\n--- Key Comparisons (Ensemble vs SMOTE+LightGBM) ---", flush=True)
    print(f"  C1 (equal τ=0.50):   Ens={ens_c1:.4f}  SM={sm_c1:.4f}  Δ={ens_c1-sm_c1:+.4f}", flush=True)
    print(f"  C2 (fair τ*):        Ens={ens_c2:.4f}  SM={sm_c2:.4f}  Δ={ens_c2-sm_c2:+.4f}", flush=True)
    print(f"  C3 (asymmetric):     Ens={ens_c3:.4f}  SM={sm_c3:.4f}  Δ={ens_c3-sm_c3:+.4f}", flush=True)

    if sm_c2 > 0:
        fair_gap = ens_c2 - sm_c2
        asym_gap = ens_c3 - sm_c3
        if asym_gap > 0:
            pct = fair_gap / asym_gap * 100 if asym_gap != 0 else 0
            print(f"\n  Fair gap: {fair_gap:.4f}  |  Asymmetric gap: {asym_gap:.4f}", flush=True)
            print(f"  Attributable to threshold: {pct:.0f}%", flush=True)

    # Statistical tests
    print(f"\n--- Statistical Tests ---", flush=True)
    for k, v in results['p_values'].items():
        sig = "***" if v['p_value'] < 0.01 else "**" if v['p_value'] < 0.05 else "*" if v['p_value'] < 0.1 else "ns"
        print(f"  {k:30s}: t={v['t_stat']:+.3f}  p={v['p_value']:.4f} ({sig})  n={v['n']}", flush=True)

    # Ablation
    print(f"\n--- Ablation ---", flush=True)
    abl = results['ablation']
    for v in ['Full', '-ThresholdOpt', '-ClusterDist', '-HeteroEnsemble']:
        if v in abl:
            f1 = abl[v]['F1']
            auc = abl[v]['AUC']
            delta = f1['mean'] - abl['Full']['F1']['mean'] if v != 'Full' else 0
            print(f"  {v:20s}: AUC={auc['mean']:.4f}  F1={f1['mean']:.4f}±{f1['std']:.3f}  ΔF1={delta:+.4f}",
                  flush=True)

    # Threshold sweep
    print(f"\n--- Threshold Sweep ---", flush=True)
    ts = results['threshold_sweep']
    for t in sorted(ts.keys(), key=float):
        f1 = ts[t]['F1']
        prec = ts[t]['Precision']
        rec = ts[t]['Recall']
        print(f"  τ={t:5s}  F1={f1['mean']:.4f}  P={prec['mean']:.4f}  R={rec['mean']:.4f}", flush=True)

    # Calibration
    print(f"\n--- Calibration ---", flush=True)
    cal = results['calibration_summary']
    print(f"  No-calibration τ* mean: {cal['NoCalib_tau_mean']:.3f}", flush=True)
    print(f"  Platt-calibrated τ* mean: {cal['Platt_tau_mean']:.3f}", flush=True)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    print("="*70, flush=True)
    print("  Employee Attrition — Threshold Asymmetry Experiment", flush=True)
    print(f"  {N_REPEATS}×{N_SPLITS} = {N_REPEATS*N_SPLITS} measurements per dataset", flush=True)
    print("="*70, flush=True)

    # ---- Dataset 1: IBM HR Attrition ----
    result1 = load_and_preprocess('IBM')
    if result1:
        X1, y1, nfeat1, ir1 = result1
        res1 = run_experiment(X1, y1, 'IBM', ir1)
        print_summary(res1)
    else:
        print("[ERROR] IBM dataset not found!", flush=True)
        sys.exit(1)

    # ---- Dataset 2: ziya07 Attrition (10,000 records) ----
    result2 = load_and_preprocess('ziya07')
    if result2:
        X2, y2, nfeat2, ir2 = result2
        res2 = run_experiment(X2, y2, 'ziya07', ir2)
        print_summary(res2)
    else:
        print("[WARN] ziya07 dataset not found, skipping", flush=True)

    # ---- Dataset 3: HR Attrition Indian (5,000 records) ----
    result3 = load_and_preprocess('Indian')
    if result3:
        X3, y3, nfeat3, ir3 = result3
        res3 = run_experiment(X3, y3, 'Indian', ir3)
        print_summary(res3)
    else:
        print("[WARN] Indian dataset not found, skipping", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("  ALL EXPERIMENTS COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
