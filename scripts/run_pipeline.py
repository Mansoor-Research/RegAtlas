#!/usr/bin/env python
"""
Machine Learning Pipeline for Genetic Variant Prioritization (Revisions Version).

This script loads integrated GWAS, eQTL, and RNA-seq data to train and evaluate
machine learning models (MLP and LightGBM) for prioritizing genetic variants.
It ensembles both models using a weighted vote (average of calibrated probabilities).
"""

# --- Section 1: Imports & Setup ---

import os
import logging
import sys
import argparse
from typing import List, Tuple, Optional, Dict, Callable

# --- Core Data Science Libraries ---
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

# --- Scikit-learn ---
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc as sk_auc,
    precision_recall_curve, average_precision_score, brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.class_weight import compute_class_weight

# --- TensorFlow / Keras ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# --- Specialized Libraries ---
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not found. Please install: pip install lightgbm", file=sys.stderr)
    sys.exit(1)

try:
    import shap
except ImportError:
    print("SHAP not found. Please install: pip install shap", file=sys.stderr)
    sys.exit(1)


# --- Global Configuration & Style ---
TARGET_PRECS = [0.10, 0.15, 0.20]  # Validation precision targets
GWAS_SIG = 5e-8                   # Genome-wide significance threshold
RANDOM_STATE = 42

sns.set_style("ticks")
sns.set_context("paper")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Section 2: Helper Functions ---

# --- Plotting & Bootstrap Helpers ---

def bootstrap_envelope(y_true: np.ndarray,
                       y_score: np.ndarray,
                       curve_xy_fn: Callable,
                       n_boot: int = 500,
                       seed: int = RANDOM_STATE,
                       x_grid: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates a 95% CI envelope for a curve (e.g., ROC, PR) via bootstrap."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    rng = np.random.RandomState(seed)
    curves = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        x, y = curve_xy_fn(y_true[idx], y_score[idx])
        curves.append((x, y))

    if x_grid is None:
        x_grid = np.linspace(0, 1, 200)

    ys = []
    for x, y in curves:
        ys.append(np.interp(x_grid, x, y))

    ys = np.vstack(ys)
    lo, hi = np.percentile(ys, [2.5, 97.5], axis=0)
    return x_grid, lo, hi

def _roc_xy(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to get (FPR, TPR) for bootstrapping."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr

def _pr_xy(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to get (Recall, Precision) for bootstrapping."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return rec, prec

def plot_roc_with_ci(y_true: np.ndarray, y_score: np.ndarray, out_prefix: str) -> float:
    """Plots ROC curve with 95% CI and saves to file."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = sk_auc(fpr, tpr)
    xs, lo, hi = bootstrap_envelope(y_true, y_score, _roc_xy, n_boot=300)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    plt.fill_between(xs, lo, hi, alpha=0.15, label='95% CI')
    plt.plot([0, 1], [0, 1], '--', lw=1, color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg')
    plt.savefig(f'{out_prefix}.png', dpi=300)
    plt.close()
    return auc

def plot_pr_with_ci(y_true: np.ndarray, y_score: np.ndarray, out_prefix: str) -> float:
    """Plots Precision-Recall curve with 95% CI and saves to file."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    xs, lo, hi = bootstrap_envelope(y_true, y_score, _pr_xy, n_boot=300)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(rec, prec, lw=2, label=f'AP = {ap:.3f}')
    plt.fill_between(xs, lo, hi, alpha=0.15, label='95% CI')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg')
    plt.savefig(f'{out_prefix}.png', dpi=300)
    plt.close()
    return ap

def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, out_prefix: str) -> float:
    """Plots calibration curve and saves to file."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=15, strategy='quantile')
    brier = brier_score_loss(y_true, y_score)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Model Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title(f'Calibration (Brier = {brier:.3f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg')
    plt.savefig(f'{out_prefix}.png', dpi=300)
    plt.close()
    return brier

def plot_confusion(cm: np.ndarray, labels: List[str], out_prefix: str, normalize: bool = True):
    """Plots a confusion matrix and saves to file."""
    m = cm.astype(float)
    if normalize:
        m = m / (m.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(
        m, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
        xticklabels=labels, yticklabels=labels, cbar=False
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}.svg')
    plt.savefig(f'{out_prefix}.png', dpi=300)
    plt.close()

# --- Metric & Evaluation Helpers ---

def pick_thresholds(val_y: np.ndarray,
                    val_proba: np.ndarray,
                    targets: List[float] = (0.10, 0.15, 0.20)
                    ) -> Tuple[float, Dict[float, Optional[float]]]:
    """Finds threshold for best F1 and for specific precision targets on validation data."""
    prec, rec, thr = precision_recall_curve(val_y, val_proba)
    prec, rec = prec[:-1], rec[:-1]

    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    thr_best_f1 = thr[best_idx]

    thr_by_prec = {}
    for T in targets:
        idx = np.where(prec >= T)[0]
        thr_by_prec[T] = (thr[idx[0]] if len(idx) > 0 else None)

    return thr_best_f1, thr_by_prec

def topk_preds(test_proba: np.ndarray,
               y_val: pd.Series
               ) -> Tuple[np.ndarray, float, int]:
    """Generates predictions by taking the top K % of test samples,
    where K is the prevalence in the validation set."""
    val_prev = y_val.mean()
    k = int(np.ceil(len(test_proba) * val_prev))
    order = np.argsort(-test_proba)
    pred = np.zeros_like(test_proba, dtype=int)
    pred[order[:k]] = 1
    return pred, val_prev, k

def summarize_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_proba: np.ndarray,
                       tag: str,
                       out_dir: str
                       ) -> Dict[str, float]:
    """Generates all plots and key metrics for a given model output."""
    logging.info(f"Summarizing metrics for: {tag}")
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = plot_roc_with_ci(y_true, y_proba, os.path.join(out_dir, f'{tag}_roc'))
    pr_auc = plot_pr_with_ci(y_true, y_proba, os.path.join(out_dir, f'{tag}_pr'))
    brier = plot_calibration(y_true, y_proba, os.path.join(out_dir, f'{tag}_calibration'))
    plot_confusion(cm, ['0', '1'], os.path.join(out_dir, f'{tag}_cmatrix'), normalize=True)

    logging.info(f"{tag} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'brier': brier}

def point_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates and prints precision, recall, and F1 for a binary prediction."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = f1_score(y_true, y_pred)
    logging.info(f"{name:>15} | P={prec:.3f} R={rec:.3f} F1={f1:.3f} | TP={tp} FP={fp} FN={fn}")
    return {'precision': prec, 'recall': rec, 'f1': f1}

def calibrate_isotonic(val_proba: np.ndarray,
                       y_val: np.ndarray,
                       test_proba: np.ndarray
                       ) -> Tuple[np.ndarray, IsotonicRegression]:
    """Trains an IsotonicRegression calibrator on validation data and applies it to test data."""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_proba, y_val)
    return ir.predict(test_proba), ir

def get_bootstrap_ci(y_true: np.ndarray,
                     y_proba: np.ndarray,
                     metric_fn: Callable,
                     B: int = 1000,
                     seed: int = RANDOM_STATE
                     ) -> Tuple[float, float]:
    """Calculates a 95% bootstrap CI for a given metric function."""
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_proba[idx]))
    return np.percentile(vals, [2.5, 97.5])


# --- Analysis & Plotting Functions ---

def run_ablation(feature_sets: Dict[str, Dict[str, List[str]]],
                 X_train_raw: pd.DataFrame, y_train: pd.Series,
                 X_val_raw: pd.DataFrame, y_val: pd.Series,
                 X_test_raw: pd.DataFrame, y_test: pd.Series,
                 out_dir: str
                 ) -> pd.DataFrame:
    """Runs an ablation study by training LGBM on different feature subsets."""
    logging.info("Starting feature ablation study...")
    abl_rows = []

    for tag, feats in feature_sets.items():
        logging.info(f"Running ablation for: {tag}")
        num_feats = feats.get('num', [])
        cat_feats = feats.get('cat', [])

        pre = ColumnTransformer([
            ('num', StandardScaler(), num_feats),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
        ], remainder='drop')

        Xtr = pre.fit_transform(X_train_raw)
        Xva = pre.transform(X_val_raw)
        Xte = pre.transform(X_test_raw)
        ytr, yva, yte = y_train.values, y_val.values, y_test.values

        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.01, num_leaves=15,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=10.0,
            min_child_samples=500, objective='binary', n_jobs=-1,
            random_state=RANDOM_STATE, scale_pos_weight=1.0
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])

        test_proba = clf.predict_proba(Xte)[:, 1]
        pr_auc = average_precision_score(yte, test_proba)
        fpr, tpr, _ = roc_curve(yte, test_proba)
        roc_auc = sk_auc(fpr, tpr)
        brier = brier_score_loss(yte, test_proba)

        _, val_prev, k = topk_preds(test_proba, y_val)
        y_pred_topk = (np.argsort(-test_proba) < k).astype(int)
        
        metrics = point_metrics(f"Ablation {tag}", yte, y_pred_topk)
        
        abl_rows.append(dict(
            model=tag, pr_auc=pr_auc, roc_auc=roc_auc, brier=brier,
            k=k, **metrics
        ))

    abl_df = pd.DataFrame(abl_rows)
    abl_path = os.path.join(out_dir, 'ablation_results.csv')
    abl_df.to_csv(abl_path, index=False)
    logging.info(f"Saved ablation results to {abl_path}")
    return abl_df


def volcano_plot(df_test_like: pd.DataFrame,
                 novel_hits: pd.DataFrame,
                 out_prefix: str):
    """Generates a volcano plot overlaying novel predictions."""
    if not {'log2FoldChange', 'padj'}.issubset(df_test_like.columns):
        logging.warning("Volcano plot skipped: missing 'log2FoldChange' or 'padj'")
        return
    
    tmp = df_test_like.copy()
    tmp['neglog10_padj'] = -np.log10(tmp['padj'].replace(0, 1e-300))

    plt.figure(figsize=(10, 6))
    plt.scatter(tmp['log2FoldChange'], tmp['neglog10_padj'], s=8, alpha=0.4, 
                color='gray', label='All TEST pairs')
    
    if len(novel_hits):
        nh_idx = tmp.index.intersection(novel_hits.index)
        nh = tmp.loc[nh_idx]
        plt.scatter(nh['log2FoldChange'], nh['neglog10_padj'], s=18, 
                    c='red', label='Novel Predictions')
    
    plt.axhline(-np.log10(0.05), color='blue', linestyle='--', label='padj = 0.05')
    plt.axvline(1, color='green', linestyle='--', label='|log2FC| = 1')
    plt.axvline(-1, color='green', linestyle='--')
    plt.xlabel('log2FoldChange')
    plt.ylabel('-log10(padj)')
    plt.title('Volcano Plot (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + '.png', dpi=300)
    plt.savefig(out_prefix + '.svg')
    plt.close()


def manhattan_overlay(df_test_like: pd.DataFrame,
                      novel_hits: pd.DataFrame,
                      out_prefix: str):
    """Generates a Manhattan plot overlaying novel predictions."""
    need = {'CHROM', 'POS', 'PVAL'}
    if not need.issubset(df_test_like.columns):
        logging.warning("Manhattan plot skipped: missing CHROM, POS, or PVAL")
        return
    
    tmp = df_test_like.copy()
    tmp['neglog10P'] = -np.log10(tmp['PVAL'].replace(0, 1e-300))

    def _chrom_to_num(c):
        s = str(c).upper().replace('CHR', '')
        return {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(s, int(s) if s.isdigit() else 0)

    tmp['CHR_NUM'] = tmp['CHROM'].map(_chrom_to_num)
    tmp = tmp.sort_values(['CHR_NUM', 'POS']).reset_index()
    tmp['ind'] = np.arange(len(tmp))

    plt.figure(figsize=(12, 6))
    plt.scatter(tmp['ind'], tmp['neglog10P'], s=4, c='lightgray', label='All TEST SNPs')
    
    if len(novel_hits):
        idx = tmp['index'].isin(novel_hits.index)
        plt.scatter(tmp.loc[idx, 'ind'], tmp.loc[idx, 'neglog10P'], s=12, 
                    c='red', label='Novel Predictions')
    
    ticks, labels = [], []
    for chrom in sorted(tmp['CHR_NUM'].unique()):
        idx = tmp.index[tmp['CHR_NUM'] == chrom]
        if len(idx):
            ticks.append((idx.min() + idx.max()) // 2)
            labels.append(str(chrom))
    
    plt.xticks(ticks, labels)
    plt.xlabel('Chromosome')
    plt.ylabel('-log10(GWAS P-value)')
    plt.title('Manhattan Plot Overlay (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + '.png', dpi=300)
    plt.savefig(out_prefix + '.svg')
    plt.close()


# --- Section 3: Main Execution ---

def run_analysis(data_csv_path: str, output_tag: str):
    OUT_DIR = f'results_{output_tag}'
    os.makedirs(OUT_DIR, exist_ok=True)
    
    logging.info(f"--- Starting pipeline for: {output_tag} ---")
    logging.info(f"Output directory: {OUT_DIR}")

    # --- 1. Load Data & Define Label ---
    
    logging.info(f"Loading data from {data_csv_path}...")
    try:
        df = pd.read_csv(data_csv_path)
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_csv_path}")
        return

    for col in ['hgnc_symbol', 'gene_biotype', 'description']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    need = {'log2FoldChange', 'padj', 'pval_nominal'}
    missing = need - set(df.columns)
    assert not missing, f'Missing columns for label definition: {missing}'

    df['label'] = (
        (df['log2FoldChange'].abs() > 1) &
        (df['padj'] < 0.05) &
        (df['pval_nominal'] < 1e-4)
    ).astype(int)

    logging.info(f"Data loaded: {df.shape} | Positives: {int(df['label'].sum())}")

    # --- 2. Feature Engineering & Selection ---
    
    logging.info("Starting feature engineering...")
    
    RISKY_LABEL_COLS = {
        'log2FoldChange', 'padj', 'pvalue', 'stat', 'pval_nominal',
        'min_pval_nominal', 'pval_nominal_threshold', 'pval_beta',
        'log2FoldChange_neglog10', 'padj_neglog10', 'pval_nominal_neglog10', 'pvalue_neglog10'
    }
    
    IDENTIFIER_COLS = {
        'CHROM', 'ID', 'POS', 'variant_id', 'gene_id', 'ensembl_gene_id',
        'hgnc_symbol', 'description', 'A1', 'A2', 'DIRE'
    }
    
    SAFE_NUM_CANDIDATES = [
        'BETA', 'SE', 'PVAL', 'NCAS', 'NCON', 'NGT', 'NEFFDIV2', 'IMPINFO', 'FCAS', 'FCON',
        'tss_distance', 'slope', 'slope_se', 'af', 'ma_samples', 'ma_count',
        'baseMean', 'lfcSE'
    ]

    if 'PVAL' in df.columns:
        df['PVAL_neglog10'] = -np.log10(df['PVAL'].replace(0, 1e-300))
        if 'PVAL_neglog10' not in SAFE_NUM_CANDIDATES:
             SAFE_NUM_CANDIDATES.append('PVAL_neglog10')

    if 'tss_distance' in df.columns:
        df['tss_bin'] = pd.cut(df['tss_distance'],
                               bins=[-np.inf, 1e3, 1e4, 1e5, 1e6, np.inf],
                               labels=['<1kb', '1-10kb', '10-100kb', '0.1-1Mb', '>1Mb'])
    if 'af' in df.columns:
        df['maf_bin'] = pd.cut(df['af'], bins=[0, 0.01, 0.05, 0.2, 0.5],
                               labels=['ultra-rare', 'rare', 'low-freq', 'common'])

    SAFE_CAT_CANDIDATES = [c for c in ['tss_bin', 'maf_bin'] if c in df.columns]

    present = set(df.columns)
    safe_num = [c for c in SAFE_NUM_CANDIDATES if c in present and c not in RISKY_LABEL_COLS]
    safe_cat = [c for c in SAFE_CAT_CANDIDATES if c in present]

    cols_for_model = safe_num + safe_cat + ['label']
    df_model = df[cols_for_model].dropna().copy()
    df_ids = df[[c for c in IDENTIFIER_COLS if c in df.columns]].loc[df_model.index].copy()

    # --- 3. Data Splitting (Grouped) ---
    
    assert 'gene_id' in df.columns, "'gene_id' column is required for grouped split."
    groups_all = df.loc[df_model.index, 'gene_id'].astype(str)

    X_all = df_model[safe_num + safe_cat]
    y_all = df_model['label'].astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss.split(X_all, y_all, groups_all))
    
    X_trainval, X_test = X_all.iloc[trainval_idx], X_all.iloc[test_idx]
    y_trainval, y_test = y_all.iloc[trainval_idx], y_all.iloc[test_idx]
    id_test = df_ids.iloc[test_idx].copy()
    groups_tv = groups_all.iloc[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE + 1)
    tr_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_tv))
    
    X_train, X_val = X_trainval.iloc[tr_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[tr_idx], y_trainval.iloc[val_idx]

    # --- 4. Preprocessing & Class Weights ---
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), safe_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), safe_cat),
    ], remainder='drop')

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    try:
        oh_names = preprocessor.named_transformers_['cat'].get_feature_names_out(safe_cat)
    except Exception:
        oh_names = []
    feature_names_proc = list(safe_num) + list(oh_names)

    cw = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train.values)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    # --- 5. Model Training ---
    
    # 5a. MLP
    logging.info("Training MLP model...")
    input_dim = X_train_proc.shape[1]
    roc_auc_metric = AUC(name='auc', curve='ROC')
    pr_auc_metric = AUC(name='pr_auc', curve='PR')

    mlp = Sequential([
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(2e-4), input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.40),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(2e-4)),
        BatchNormalization(),
        Dropout(0.40),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(2e-4)),
        Dropout(0.30),
        Dense(1, activation='sigmoid')
    ])

    mlp.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy', roc_auc_metric, pr_auc_metric])

    es = EarlyStopping(monitor='val_pr_auc', mode='max', patience=6, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    mlp.fit(X_train_proc, y_train,
            validation_data=(X_val_proc, y_val),
            epochs=100,
            batch_size=1024,
            class_weight=class_weight,
            callbacks=[es, rlr],
            verbose=1)

    val_proba_mlp = mlp.predict(X_val_proc, batch_size=4096).ravel()
    test_proba_mlp = mlp.predict(X_test_proc, batch_size=4096).ravel()

    # 5b. LightGBM (Regularized to prevent 1-tree early stopping)
    logging.info("Training LightGBM model with logloss early stopping...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.01, num_leaves=15,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=10.0,
        min_child_samples=500, objective='binary', n_jobs=-1,
        random_state=RANDOM_STATE, scale_pos_weight=1.0
    )

    lgbm.fit(X_train_proc, y_train,
             eval_set=[(X_val_proc, y_val)],
             eval_metric='binary_logloss',
             callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)])

    val_proba_lgbm = lgbm.predict_proba(X_val_proc)[:, 1]
    test_proba_lgbm = lgbm.predict_proba(X_test_proc)[:, 1]

    # --- 6. Calibration & MLP + LightGBM Weighted Vote Ensembling ---
    
    logging.info("Calibrating model probabilities...")

    # Calibrate MLP using Isotonic Regression
    test_proba_mlp_cal, iso_mlp = calibrate_isotonic(val_proba_mlp, y_val.values, test_proba_mlp)
    val_proba_mlp_cal = iso_mlp.predict(val_proba_mlp)

    # Calibrate LGBM using Isotonic Regression
    test_proba_lgbm_cal, iso_lgbm = calibrate_isotonic(val_proba_lgbm, y_val.values, test_proba_lgbm)
    val_proba_lgbm_cal = iso_lgbm.predict(val_proba_lgbm)

    # Define Primary Model via Weighted Vote Mechanism (average of calibrated MLP and LightGBM)
    PROBS_TEST_PRIMARY = 0.5 * test_proba_lgbm_cal + 0.5 * test_proba_mlp_cal
    PROBS_VAL_PRIMARY = 0.5 * val_proba_lgbm_cal + 0.5 * val_proba_mlp_cal
    
    logging.info("Primary model selected: Calibrated MLP + LightGBM Weighted Vote")

    y_pred_primary, val_prev_primary, k_primary = topk_preds(PROBS_TEST_PRIMARY, y_val)
    logging.info(f"Primary operating point (top-K): {k_primary} predictions ({val_prev_primary:.3%})")

    # --- 7. Core Evaluation & Interpretability ---
    
    metrics = summarize_metrics(
        y_test.values, y_pred_primary, PROBS_TEST_PRIMARY, 
        'primary_weighted_vote_topK', OUT_DIR
    )
    
    with open(os.path.join(OUT_DIR, 'metrics_primary.txt'), 'w') as f:
        f.write('=== PRIMARY MODEL (MLP + LightGBM Weighted Vote) ===\n')
        f.write(f'Operating point: top-K by validation prevalence ({val_prev_primary:.3%})\n')
        f.write(classification_report(y_test, y_pred_primary, digits=3))
        f.write("\n--- Overall Metrics ---\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR-AUC:  {metrics['pr_auc']:.4f}\n")
        f.write(f"Brier:   {metrics['brier']:.4f}\n")
    
    # Point Metrics Comparison
    logging.info("--- Point Metrics at Top-K Operating Point ---")
    point_metrics("MLP (raw)", y_test.values, topk_preds(test_proba_mlp, y_val)[0])
    point_metrics("MLP (cal)", y_test.values, topk_preds(test_proba_mlp_cal, y_val)[0])
    point_metrics("LGBM (raw)", y_test.values, topk_preds(test_proba_lgbm, y_val)[0])
    point_metrics("LGBM (cal)", y_test.values, topk_preds(test_proba_lgbm_cal, y_val)[0])
    point_metrics("Weighted Vote", y_test.values, y_pred_primary)
    
    logging.info("Calculating 95% CIs for primary model...")
    ap_lo, ap_hi = get_bootstrap_ci(y_test.values, PROBS_TEST_PRIMARY, average_precision_score)
    roc_lo, roc_hi = get_bootstrap_ci(y_test.values, PROBS_TEST_PRIMARY, 
                                    lambda y, p: sk_auc(*roc_curve(y, p)[:2]))
    logging.info(f"PR-AUC 95% CI: [{ap_lo:.3f}, {ap_hi:.3f}]")
    logging.info(f"ROC-AUC 95% CI: [{roc_lo:.3f}, {roc_hi:.3f}]")

    # Precision@K Table (Now showing realistic curves, no ties flatline)
    logging.info("Calculating Precision@K table...")
    def precision_at_frac(frac):
        k = max(1, int(np.ceil(len(PROBS_TEST_PRIMARY) * frac)))
        order = np.argsort(-PROBS_TEST_PRIMARY)
        sel = np.zeros_like(PROBS_TEST_PRIMARY, dtype=int); sel[order[:k]] = 1
        metrics = point_metrics(f"P@{100*frac:.2f}% (k={k})", y_test.values, sel)
        return dict(K=f"{100*frac:.2f}%", k=k, **metrics)

    prec_at_k_rows = [precision_at_frac(frac) for frac in [0.005, 0.01, 0.015, 0.02]]
    prec_at_k_df = pd.DataFrame(prec_at_k_rows)
    prec_at_k_df.to_csv(os.path.join(OUT_DIR, 'precision_at_k.csv'), index=False)
    logging.info(f"Precision@K table:\n{prec_at_k_df}")

    # SHAP Interpretability on the LightGBM branch
    logging.info("Running SHAP analysis on LightGBM branch...")
    explainer = shap.TreeExplainer(lgbm, feature_names=feature_names_proc)
    X_test_sample = shap.sample(X_test_proc, 1000, random_state=RANDOM_STATE)
    shap_values = explainer(X_test_sample)

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=10)
    plt.title("SHAP Feature Importance (LGBM branch)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_lgbm_bar.svg"))
    plt.savefig(os.path.join(OUT_DIR, "shap_lgbm_bar.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=10)
    plt.title("SHAP Beeswarm Plot (LGBM branch)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_lgbm_beeswarm.svg"))
    plt.savefig(os.path.join(OUT_DIR, "shap_lgbm_beeswarm.png"), dpi=300)
    plt.close()

    # --- 8. Downstream Analysis: Ablation & Novel Hits ---

    # Ablation Study
    present_cols = set(df_model.columns)
    gwas_num = [c for c in ['BETA', 'SE', 'PVAL_neglog10', 'NCAS', 'NCON', 'IMPINFO', 'af', 'tss_distance'] if c in present_cols]
    eqtl_num = [c for c in ['slope', 'slope_se'] if c in present_cols]
    rnaseq_num = [c for c in ['baseMean', 'lfcSE'] if c in present_cols]
    base_cat = [c for c in ['tss_bin', 'maf_bin'] if c in present_cols]

    feature_sets = {
        "GWAS-only": {"num": gwas_num, "cat": base_cat},
        "GWAS+eQTL": {"num": gwas_num + eqtl_num, "cat": base_cat},
        "GWAS+eQTL+RNA-seq": {"num": gwas_num + eqtl_num + rnaseq_num, "cat": base_cat},
        "ALL-safe (primary)": {"num": safe_num, "cat": safe_cat}
    }
    
    abl_df = run_ablation(
        feature_sets, X_train, y_train, X_val, y_val, X_test, y_test, OUT_DIR
    )

    # Generate Novel Hit List
    logging.info("Generating 'novel hit' list...")
    strict_frac = 0.005 
    k_strict = max(1, int(np.ceil(len(PROBS_TEST_PRIMARY) * strict_frac)))
    order = np.argsort(-PROBS_TEST_PRIMARY)
    sel = np.zeros_like(PROBS_TEST_PRIMARY, dtype=bool)
    sel[order[:k_strict]] = True

    novel = df_ids.loc[X_test.index].copy()
    novel['prob_primary'] = PROBS_TEST_PRIMARY
    novel['selected_topK_strict'] = sel.astype(int)

    for c in ['PVAL', 'IMPINFO', 'pval_nominal', 'log2FoldChange', 'padj', 'tss_distance', 'af', 'slope', 'baseMean']:
        if c in df.columns:
            novel[c] = df.loc[novel.index, c].values

    mask = (
        (novel['selected_topK_strict'] == 1)
        & (novel.get('IMPINFO', 1.0) >= 0.90)
        & (novel.get('PVAL', 1.0) > GWAS_SIG)
    )
    if 'tss_distance' in novel.columns:
        mask &= (novel['tss_distance'].abs() <= 100_000)

    novel_hits = novel.loc[mask].sort_values('prob_primary', ascending=False)
    
    out_novel = os.path.join(OUT_DIR, f'novel_hits_top{strict_frac*100:.1f}pct.csv')
    novel_hits.to_csv(out_novel, index=False)
    logging.info(f"Saved {len(novel_hits)} novel candidates to {out_novel}")

    # --- 9. Publication Figure Generation (Corrected Plotting Bug) ---
    
    logging.info("Generating publication figures...")
    
    y_true_fig = y_test.values
    y_score_fig = PROBS_TEST_PRIMARY
    y_pred_fig = y_pred_primary

    palette = sns.color_palette("colorblind")
    c1 = palette[0] # Blue
    c2 = palette[4] # Purple
    c3 = palette[2] # Green

    fpr, tpr, _ = roc_curve(y_true_fig, y_score_fig)
    roc_auc = sk_auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_fig, y_score_fig)
    pr_auc = average_precision_score(y_true_fig, y_score_fig)
    prevalence = y_true_fig.mean()
    cm = confusion_matrix(y_true_fig, y_pred_fig)
    tn, fp, fn, tp = cm.ravel()
    op_precision = tp / (tp + fp + 1e-12)
    op_recall_tpr = tp / (tp + fn + 1e-12)
    op_fpr = fp / (fp + tn + 1e-12)

    # 9b. Main 2x2 Figure (ROC + PR + Calibration + Threshold)
    sns.set_style("ticks")
    sns.set_context("talk")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 11))

    # Panel A: ROC Curve (Dot sits exactly on the curve)
    ax1.plot(fpr, tpr, lw=3, color=c1, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], '--', color='gray', lw=2, label='No Skill')
    ax1.scatter(op_fpr, op_recall_tpr, s=80, color='black', zorder=5, label='Operating Point')
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.set_title('A. ROC Curve')
    ax1.legend(loc='lower right', frameon=False)
    ax1.set_aspect('equal')

    # Panel B: Precision-Recall Curve (Dot sits exactly on the curve)
    ax2.plot(rec, prec, lw=3, color=c2, label=f'PR-AUC (AP) = {pr_auc:.3f}')
    ax2.axhline(prevalence, ls='--', color='gray', lw=2, label=f'Prevalence = {prevalence:.3f}')
    ax2.scatter(op_recall_tpr, op_precision, s=80, color='black', zorder=5, label='Operating Point')
    ax2.set_xlabel('Recall (TPR)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.set_title('B. Precision-Recall Curve')
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(0.0, 1.0)

    # Panel C: Calibration (Quantile strategy for smooth bins)
    prob_true, prob_pred = calibration_curve(y_true_fig, y_score_fig, n_bins=15, strategy='quantile')
    brier = brier_score_loss(y_true_fig, y_score_fig)
    ax3.plot(prob_pred, prob_true, marker='o', ms=7, lw=3, color=c3, label='Model Calibration')
    ax3.plot([0, 1], [0, 1], '--', color='gray', lw=2, label='Perfect Calibration')
    ax3.set_title(f'C. Calibration (Brier = {brier:.3f})')
    ax3.set_xlabel('Predicted Probability')
    ax3.set_ylabel('Observed Frequency')
    ax3.legend(loc='upper left', frameon=False)

    # Panel D: Precision / Recall / F1 vs Threshold
    tau_grid = np.linspace(0.0, 0.5, 201)
    prec_g, rec_g = [], []
    for t in tau_grid:
        yhat = (y_score_fig >= t).astype(int)
        tp_ = np.sum((yhat == 1) & (y_true_fig == 1))
        fp_ = np.sum((yhat == 1) & (y_true_fig == 0))
        fn_ = np.sum((yhat == 0) & (y_true_fig == 1))
        pr_ = tp_ / (tp_ + fp_ + 1e-12)
        rc_ = tp_ / (tp_ + fn_ + 1e-12)
        prec_g.append(pr_)
        rec_g.append(rc_)
    prec_g = np.array(prec_g)
    rec_g = np.array(rec_g)
    f1_g = 2 * prec_g * rec_g / (prec_g + rec_g + 1e-12)
    
    if y_pred_fig.sum() > 0:
        tau = float(np.min(y_score_fig[y_pred_fig == 1]))
    else:
        tau = float('nan')

    ax4.plot(tau_grid, prec_g, label='Precision', lw=3, color=c2)
    ax4.plot(tau_grid, rec_g,  label='Recall',    lw=3, color=c1)
    ax4.plot(tau_grid, f1_g,   label='F1',        lw=3, color=c3)
    ax4.axvline(tau, color='black', ls=':', lw=2)
    ax4.text(tau + 0.005, 0.95, f'Chosen τ ≈ {tau:.3f}', rotation=270, va='top', ha='left', fontsize=12, color='black')
    ax4.set_title('D. Metrics vs. Probability Threshold (\u03c4)')
    ax4.set_xlabel('Probability Threshold (\u03c4)')
    ax4.set_ylabel('Metric Value')
    ax4.legend(loc='upper right', frameon=False)
    ax4.set_xlim(0.0, 0.20)

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(OUT_DIR, 'main_figure_1x2_ROC_PR.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'main_figure_1x2_ROC_PR.svg'))
    plt.savefig(os.path.join(OUT_DIR, 'diagnostic_figure_2x2_CLEAN_v2_colors.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'diagnostic_figure_2x2_CLEAN_v2_colors.svg'))
    plt.close()

    # 9c. Combined 1x2 Figure (Ablation + SHAP)
    fig, (ax_abl, ax_shap) = plt.subplots(1, 2, figsize=(16, 6), width_ratios=[2, 3])
    
    ablation_colors = sns.color_palette("viridis", n_colors=len(abl_df))
    y_pos = np.arange(len(abl_df['model']))
    ax_abl.barh(y_pos, abl_df['pr_auc'], color=ablation_colors)
    ax_abl.set_yticks(y_pos)
    ax_abl.set_yticklabels(abl_df['model'])
    ax_abl.set_xlabel('PR-AUC (TEST)')
    ax_abl.set_title('A. Ablation: Feature Contribution')
    ax_abl.set_xlim(left=0)
    ax_abl.invert_yaxis()
    for i, v in enumerate(abl_df['pr_auc']):
        ax_abl.text(v + 0.005, i, f'{v:.3f}', va='center', color='black', fontsize=12)
    sns.despine(ax=ax_abl)

    plt.sca(ax_shap)
    shap.plots.beeswarm(shap_values, max_display=10, show=False, plot_size=None)
    ax_shap.set_title('B. SHAP Beeswarm Plot (Top 10 Features)')
    ax_shap.set_xlabel('SHAP value (impact on model output)')
    
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(OUT_DIR, 'combined_feature_importance_1x2.png'), dpi=300)
    plt.savefig(os.path.join(OUT_DIR, 'combined_feature_importance_1x2.svg'))
    plt.close()

    # 9d. Volcano & Manhattan Overlays
    df_test_like = df.loc[X_test.index, :].copy()
    volcano_plot(df_test_like, novel_hits, os.path.join(OUT_DIR, 'volcano_test_overlay'))
    manhattan_overlay(df_test_like, novel_hits, os.path.join(OUT_DIR, 'manhattan_test_overlay'))

    logging.info("All figures generated.")
    sns.set_style("ticks")
    sns.set_context("paper")

    # --- 10. Final Data Export ---
    
    logging.info("Exporting final predictions...")
    
    test_results = df_ids.loc[X_test.index].copy()
    test_results['True_Label'] = y_test.values
    test_results['Primary_Prob'] = PROBS_TEST_PRIMARY
    test_results['Primary_Pred_TopK'] = y_pred_primary
    test_results['Prob_MLP_raw'] = test_proba_mlp
    test_results['Prob_MLP_cal'] = test_proba_mlp_cal
    test_results['Prob_LGBM_raw'] = test_proba_lgbm
    test_results['Prob_LGBM_cal'] = test_proba_lgbm_cal
    
    for c in ['PVAL', 'pval_nominal', 'log2FoldChange', 'padj']:
        if c in df.columns:
            test_results[c] = df.loc[test_results.index, c].values

    out_path_full = os.path.join(OUT_DIR, 'test_results_full.csv')
    test_results.to_csv(out_path_full, index=False)
    logging.info(f"Saved full test set predictions to {out_path_full}")

    out_path_simple = os.path.join(OUT_DIR, f'test_probs_{output_tag}_primary.csv')
    simple_export_df = pd.DataFrame({
        'True_Label': y_test.values,
        'Predicted_Prob': PROBS_TEST_PRIMARY
    })
    simple_export_df.to_csv(out_path_simple, index=False)
    logging.info(f"Saved simple (Label, Prob) export to {out_path_simple}")

    logging.info(f"--- Pipeline for {output_tag} finished successfully. ---")


# --- Script Entry Point ---

if __name__ == "__main__":
    if ('ipykernel' in sys.modules) and not ('-i' in sys.argv or '--input' in sys.argv):
        # Running inside a Jupyter notebook: use default paths for Schizophrenia
        logging.info("Running inside Jupyter Notebook. Using default Schizophrenia paths.")
        default_input = r"d:\Complile_project\Output\feb_26\rna_seq_eQTLs_gwas_schizophrenia.csv"
        default_tag = "SCZ"
        run_analysis(data_csv_path=default_input, output_tag=default_tag)
    else:
        # Running as a command-line script
        parser = argparse.ArgumentParser(
            description="Run the ML pipeline for genetic variant prioritization."
        )
        parser.add_argument(
            "-i", "--input",
            type=str,
            required=True,
            help="Path to the input data CSV file"
        )
        parser.add_argument(
            "-o", "--output_tag",
            type=str,
            required=True,
            help="Short name for the output folder"
        )
        args = parser.parse_args()
        run_analysis(data_csv_path=args.input, output_tag=args.output_tag)

