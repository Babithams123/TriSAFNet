"""
03_run_experiments.py
Main experiment runner. Trains all models with spatial cross-validation,
computes all metrics, runs benchmarks, generates all plots and tables
needed for the revised paper.

Usage:
    python scripts/03_run_experiments.py --year 2018
    python scripts/03_run_experiments.py --year 2024
    python scripts/03_run_experiments.py --year both
    python scripts/03_run_experiments.py --year 2018 --ablation-trisafnet

Outputs (in timestamped run directories under ./runs/):
    - results_YYYY.csv, cv_folds_YYYY.csv
    - confusion matrices, training curves, feature importance
    - model_comparison, benchmark tables, per-class metrics
    - McNemar / Friedman / Nemenyi statistical tests
    - Ablation results (if --ablation-trisafnet)
    - All trained models in models/ subfolder
    - Structured logs in logs/ subfolder
"""

import os, sys
_dll_dir = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.isdir(_dll_dir):
    os.environ['PATH'] = _dll_dir + os.pathsep + os.environ.get('PATH', '')
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_dll_dir)
_cuda_data_dir = os.path.join(sys.prefix, 'Library')
if os.path.isfile(os.path.join(_cuda_data_dir, 'bin', 'libdevice.10.bc')):
    os.environ.setdefault('XLA_FLAGS', f'--xla_gpu_cuda_data_dir={_cuda_data_dir}')

import argparse
import gc
import json
import logging
import shutil
import time
import traceback
import tracemalloc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import chi2, friedmanchisquare

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *
import configs.config as _cfg_module
from models import (
    TriSAFNet,
    LightweightCNNRF, CNNOnly,
    LightGBMCNNRF,
    MobileNetV2Classifier,
    RFOnly, XGBoostOnly,
    EfficientNetB0Classifier, CompactViTClassifier,
    build_lightweight_cnn_feature_extractor,
)
from utils.mlflow_utils import (
    init_mlflow, start_run, end_run, log_config, log_metrics,
    log_artifact_safe, log_artifacts_dir,
    build_artifact_manifest,
)
import mlflow

import tensorflow as tf
tf.get_logger().setLevel('WARNING')

_ACTIVE_PATCH_SIZE = PATCH_SIZE
_ACTIVE_CLASS_IDS = list(range(len(CLASS_NAMES)))
_ACTIVE_CLASS_NAMES = list(CLASS_NAMES)

logger = logging.getLogger(__name__)


def class_name_from_id(class_id):
    class_id = int(class_id)
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"Class {class_id}"


def active_class_names():
    return [class_name_from_id(c) for c in _ACTIVE_CLASS_IDS]


def sync_active_classes(y):
    """Update runtime class IDs/names from the current dataset."""
    global _ACTIVE_CLASS_IDS, _ACTIVE_CLASS_NAMES
    _ACTIVE_CLASS_IDS = [int(c) for c in np.unique(y)]
    _ACTIVE_CLASS_NAMES = active_class_names()


def _safe_load_object_array(data, key, fallback):
    """Load a potentially-pickled NPZ field, returning *fallback* on error."""
    if key not in data.files:
        return fallback
    try:
        return data[key].tolist()
    except (ModuleNotFoundError, ImportError):
        return fallback


def load_npz_dataset(npz_path):
    """Load dataset NPZ and return dict of arrays + metadata.
    The NpzFile handle is closed after extraction to release the file descriptor."""
    data = np.load(npz_path, allow_pickle=True)
    patches = np.array(data['patches']).astype('float32', copy=False)  # halve RAM vs float64
    labels = np.array(data['labels'])
    n_bands = int(patches.shape[-1])
    band_names = _safe_load_object_array(
        data, 'band_names',
        BAND_NAMES[:n_bands] if n_bands <= len(BAND_NAMES) else [f"band_{i+1:02d}" for i in range(n_bands)]
    )
    patch_size = int(data['patch_size']) if 'patch_size' in data.files else int(patches.shape[1])
    class_ids = _safe_load_object_array(
        data, 'class_ids',
        [int(c) for c in np.unique(labels)]
    )
    block_ids = np.array(data['block_ids']) if 'block_ids' in data.files else None
    scaler_min = np.array(data['scaler_min']) if 'scaler_min' in data.files else None
    scaler_max = np.array(data['scaler_max']) if 'scaler_max' in data.files else None
    data.close()

    arrays = {
        'patches': patches, 'labels': labels,
        'block_ids': block_ids,
        'scaler_min': scaler_min, 'scaler_max': scaler_max,
    }
    meta = {
        'n_bands': n_bands,
        'patch_size': patch_size,
        'band_names': band_names,
        'class_ids': class_ids,
    }
    return arrays, meta


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(run_dir):
    """Configure console + file logging for the experiment run."""
    logs_dir = Path(run_dir) / 'logs'
    logs_dir.mkdir(exist_ok=True, parents=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(str(logs_dir / 'experiment.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logs_dir


# ============================================================
# PLOT STYLE
# ============================================================

def setup_plot_style():
    """Set publication-quality plot defaults."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        try:
            plt.style.use('seaborn-paper')
        except OSError:
            pass
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
    })


def _history_dict(history):
    """Extract dict from Keras History object or passthrough dict."""
    if history is None:
        return None
    return history.history if hasattr(history, 'history') else history


def _save_fig(fig, path):
    """Save figure as both PNG and PDF, auto-stripping known extensions."""
    p = str(path)
    for ext in ('.png', '.pdf', '.jpeg', '.jpg'):
        if p.lower().endswith(ext):
            p = p[:-len(ext)]
            break
    fig.savefig(f'{p}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{p}.pdf', bbox_inches='tight')
    plt.close(fig)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, class_names=None):
    """Compute all classification metrics."""
    if class_names is None:
        class_names = _ACTIVE_CLASS_NAMES
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=_ACTIVE_CLASS_IDS, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', labels=_ACTIVE_CLASS_IDS, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', labels=_ACTIVE_CLASS_IDS, zero_division=0)
    report = classification_report(y_true, y_pred, labels=_ACTIVE_CLASS_IDS, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=_ACTIVE_CLASS_IDS)
    return {
        'accuracy': acc, 'kappa': kappa, 'f1_macro': f1_macro,
        'precision_macro': prec_macro, 'recall_macro': rec_macro,
        'per_class': report, 'confusion_matrix': cm
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test comparing two classifiers."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b01 = np.sum(correct_a & ~correct_b)
    b10 = np.sum(~correct_a & correct_b)
    if b01 + b10 == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


# ============================================================
# CROSS-VALIDATION SPLITS
# ============================================================

def spatial_cv_splits(block_ids, n_folds=N_FOLDS):
    unique_blocks = np.unique(block_ids)
    for test_block in unique_blocks[:n_folds]:
        test_idx = np.where(block_ids == test_block)[0]
        train_idx = np.where(block_ids != test_block)[0]
        yield train_idx, test_idx


def standard_cv_splits(labels, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        yield train_idx, test_idx


# ============================================================
# EPOCH LOGGING HELPER
# ============================================================

def save_epoch_log(history, model_name, fold, logs_dir):
    """Save per-epoch training metrics from Keras History."""
    if history is None:
        return
    h = _history_dict(history)
    rows = []
    n_epochs = len(h.get('loss', []))
    for ep in range(n_epochs):
        row = {'model': model_name, 'fold': fold, 'epoch': ep + 1}
        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr']:
            if key in h:
                row[key] = h[key][ep] if ep < len(h[key]) else None
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        path = Path(logs_dir) / f'epoch_log_{safe_name}.csv'
        if path.exists():
            df.to_csv(str(path), mode='a', header=False, index=False)
        else:
            df.to_csv(str(path), index=False)


# ============================================================
# MODEL SAVING
# ============================================================

def save_model_artifact(model, model_name, models_dir):
    """Save model artifacts: .keras for DL, .joblib for ML, both for hybrids."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')

    try:
        if hasattr(model, 'model') and hasattr(model.model, 'save'):
            model.model.save(str(models_dir / f'{safe_name}.keras'))

        if hasattr(model, 'rf'):
            joblib.dump(model.rf, str(models_dir / f'{safe_name}_rf.joblib'))
        if hasattr(model, 'lgbm') and model.lgbm is not None:
            joblib.dump(model.lgbm, str(models_dir / f'{safe_name}_lgbm.joblib'))
        if hasattr(model, 'xgb') and model.xgb is not None:
            joblib.dump(model.xgb, str(models_dir / f'{safe_name}_xgb.joblib'))

        if hasattr(model, 'cnn') and hasattr(model.cnn, 'save'):
            model.cnn.save(str(models_dir / f'{safe_name}_cnn_backbone.keras'))

        if model_name == 'RF Only' and hasattr(model, 'rf'):
            joblib.dump(model.rf, str(models_dir / f'{safe_name}.joblib'))
        elif model_name == 'XGBoost Only' and hasattr(model, 'xgb') and model.xgb is not None:
            joblib.dump(model.xgb, str(models_dir / f'{safe_name}.joblib'))

        logger.info(f"  Saved model artifact: {safe_name}")
    except Exception as e:
        logger.warning(f"  Could not save {safe_name}: {e}")


# ============================================================
# RF SENSITIVITY ANALYSIS
# ============================================================

def rf_sensitivity_analysis(X_train, y_train, X_test, y_test, input_shape, output_dir):
    """Test RF performance with varying n_estimators."""
    logger.info("RF Tree Count Sensitivity Analysis")
    n_trees_list = [25, 50, 75, 100, 125, 150, 200, 300, 500]
    results = []

    cnn = build_lightweight_cnn_feature_extractor(input_shape)
    out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(cnn.output)
    trainer = tf.keras.Model(cnn.input, out)
    trainer.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    trainer.fit(X_train, y_train, epochs=30, batch_size=128, verbose=0)

    feats_train = cnn.predict(X_train, batch_size=256)
    feats_test = cnn.predict(X_test, batch_size=256)

    del trainer
    tf.keras.backend.clear_session()
    gc.collect()

    for n_trees in n_trees_list:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=15,
                                     n_jobs=-1, random_state=RANDOM_SEED)
        rf.fit(feats_train, y_train)
        preds = rf.predict(feats_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        kappa = cohen_kappa_score(y_test, preds)
        results.append({'n_estimators': n_trees, 'accuracy': acc, 'f1': f1, 'kappa': kappa})
        logger.info(f"  n_trees={n_trees}: acc={acc:.4f}, f1={f1:.4f}, kappa={kappa:.4f}")

    del cnn, feats_train, feats_test
    gc.collect()

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df['n_estimators'], df['accuracy'], 'o-', label='Accuracy', linewidth=1.5)
    ax.plot(df['n_estimators'], df['f1'], 's-', label='F1 Score', linewidth=1.5)
    ax.plot(df['n_estimators'], df['kappa'], '^-', label='Kappa', linewidth=1.5)
    ax.set_xlabel('Number of RF Trees')
    ax.set_ylabel('Score')
    ax.set_title('RF Sensitivity to Number of Trees')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f"{output_dir}/rf_sensitivity")

    try:
        with start_run(run_name="rf_sensitivity", nested=True):
            mlflow.set_tag('task', 'rf_sensitivity')
            for _, row in df.iterrows():
                log_metrics({
                    f'rf_{int(row["n_estimators"])}_acc': row['accuracy'],
                    f'rf_{int(row["n_estimators"])}_f1': row['f1'],
                    f'rf_{int(row["n_estimators"])}_kappa': row['kappa'],
                })
            log_artifact_safe(f"{output_dir}/rf_sensitivity.png")
            log_artifact_safe(f"{output_dir}/rf_sensitivity.pdf")
    except Exception as e:
        logger.warning(f"  MLflow rf_sensitivity logging failed: {e}")

    return df


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_confusion_matrix(cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    _save_fig(fig, output_path)


def plot_training_curves(history, model_name, output_path):
    h = _history_dict(history)
    if h is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(h['loss'], label='Train Loss', linewidth=1.2)
    if 'val_loss' in h:
        ax1.plot(h['val_loss'], label='Val Loss', linewidth=1.2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(h['accuracy'], label='Train Acc', linewidth=1.2)
    if 'val_accuracy' in h:
        ax2.plot(h['val_accuracy'], label='Val Acc', linewidth=1.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _save_fig(fig, output_path)


def plot_feature_importance(importances, model_name, output_path, top_n=20):
    n_feats = len(importances)
    feat_names = [f"CNN_feat_{i}" for i in range(n_feats)]
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(idx)), importances[idx], color='steelblue')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feat_names[i] for i in idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} — RF Feature Importance (Top {top_n})')
    ax.invert_yaxis()
    _save_fig(fig, output_path)


def plot_comparison_chart(results_df, year, output_path):
    metrics = ['accuracy', 'kappa', 'f1_macro']
    labels = ['Accuracy', 'Kappa', 'F1 Score']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results_df))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * width, results_df[metric], width, label=label)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Comparison — {year}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df['model'], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    _save_fig(fig, output_path)


def plot_all_confusion_matrices(all_cms, year, output_dir):
    n = len(all_cms)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (model_name, cm) in enumerate(all_cms.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=_ACTIVE_CLASS_NAMES, yticklabels=_ACTIVE_CLASS_NAMES, ax=ax)
        ax.set_title(model_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Confusion Matrices — All Models ({year})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, f'{output_dir}/all_confusion_matrices_{year}')


def plot_all_training_curves(all_histories, year, output_dir):
    models_with_history = {k: v for k, v in all_histories.items() if v is not None}
    n = len(models_with_history)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (name, hist) in enumerate(models_with_history.items()):
        ax = axes[idx]
        h = _history_dict(hist)
        ax.plot(h['loss'], label='Train', linewidth=1.2)
        if 'val_loss' in h:
            ax.plot(h['val_loss'], label='Val', linewidth=1.2)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(f'Training & Validation Loss — All Models ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, f'{output_dir}/all_loss_curves_{year}')

    fig2, axes2 = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes2 = np.array(axes2).flatten() if n > 1 else [axes2]
    for idx, (name, hist) in enumerate(models_with_history.items()):
        ax = axes2[idx]
        h = _history_dict(hist)
        ax.plot(h['accuracy'], label='Train', linewidth=1.2)
        if 'val_accuracy' in h:
            ax.plot(h['val_accuracy'], label='Val', linewidth=1.2)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Accuracy', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    for idx in range(n, len(axes2)):
        axes2[idx].axis('off')
    fig2.suptitle(f'Training & Validation Accuracy — All Models ({year})',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig2, f'{output_dir}/all_accuracy_curves_{year}')


def plot_proposed_model_detail(history, cm, y_true, y_pred, year, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    h = _history_dict(history)
    if h is not None:
        ax.plot(h['loss'], label='Train Loss', linewidth=1.5, color='#2196F3')
        if 'val_loss' in h:
            ax.plot(h['val_loss'], label='Val Loss', linewidth=1.5, color='#F44336')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('TriSAFNet — Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training history', ha='center', va='center')

    ax = axes[0, 1]
    h = _history_dict(history)
    if h is not None:
        ax.plot(h['accuracy'], label='Train Acc', linewidth=1.5, color='#4CAF50')
        if 'val_accuracy' in h:
            ax.plot(h['val_accuracy'], label='Val Acc', linewidth=1.5, color='#FF9800')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('TriSAFNet — Training & Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training history', ha='center', va='center')

    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=_ACTIVE_CLASS_NAMES, yticklabels=_ACTIVE_CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('TriSAFNet — Confusion Matrix')

    ax = axes[1, 1]
    report = classification_report(y_true, y_pred, labels=_ACTIVE_CLASS_IDS, target_names=_ACTIVE_CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    precs = [report[c]['precision'] for c in _ACTIVE_CLASS_NAMES]
    recs = [report[c]['recall'] for c in _ACTIVE_CLASS_NAMES]
    f1s = [report[c]['f1-score'] for c in _ACTIVE_CLASS_NAMES]
    x_pos = np.arange(len(_ACTIVE_CLASS_NAMES))
    w = 0.25
    ax.bar(x_pos - w, precs, w, label='Precision', color='#2196F3')
    ax.bar(x_pos, recs, w, label='Recall', color='#4CAF50')
    ax.bar(x_pos + w, f1s, w, label='F1', color='#FF9800')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(_ACTIVE_CLASS_NAMES, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('TriSAFNet — Per-Class Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'TriSAFNet (Proposed) — Detailed Performance ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, f'{output_dir}/proposed_TriSAFNet_detail_{year}')


def plot_per_class_all_models(all_predictions, year, output_dir):
    if not all_predictions:
        return
    model_names = list(all_predictions.keys())
    f1_matrix = []
    for name in model_names:
        y_t = all_predictions[name]['y_true']
        y_p = all_predictions[name]['y_pred']
        report = classification_report(y_t, y_p, labels=_ACTIVE_CLASS_IDS, target_names=_ACTIVE_CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        f1_matrix.append([report[c]['f1-score'] for c in _ACTIVE_CLASS_NAMES])

    f1_df = pd.DataFrame(f1_matrix, index=model_names, columns=_ACTIVE_CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, max(5, len(model_names) * 0.6)))
    sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='YlGn', ax=ax,
                vmin=0.5, vmax=1.0, linewidths=0.5)
    ax.set_title(f'Per-Class F1 Score — All Models ({year})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model')
    _save_fig(fig, f'{output_dir}/per_class_f1_heatmap_{year}')


def plot_sample_predictions(model, X_test, y_test, year, output_dir, n_samples=16):
    try:
        proba = model.predict_proba(X_test)
        preds = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)

        idx = np.random.RandomState(RANDOM_SEED).choice(len(X_test),
                                                         size=min(n_samples, len(X_test)),
                                                         replace=False)
        cols = 4
        rows = (len(idx) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols * 2, figsize=(3 * cols * 2, 3.5 * rows))
        axes = np.array(axes).flatten()

        for i in range(len(idx)):
            j = idx[i]
            ax_rgb = axes[i * 2]
            ax_sar = axes[i * 2 + 1]

            rgb = X_test[j][:, :, [4, 3, 2]]
            rgb = np.clip(rgb, 0, 1)
            ax_rgb.imshow(rgb)
            true_cls = class_name_from_id(y_test[j])
            pred_cls = class_name_from_id(preds[j])
            conf = confidences[j]
            color = 'green' if preds[j] == int(y_test[j]) else 'red'
            ax_rgb.set_title(f'RGB\nT:{true_cls}\nP:{pred_cls} ({conf:.0%})',
                             fontsize=6, color=color)
            ax_rgb.axis('off')

            sar = X_test[j][:, :, :3]
            sar = np.clip(sar, 0, 1)
            ax_sar.imshow(sar)
            ax_sar.set_title('SAR (VV/VH/RVI)', fontsize=6)
            ax_sar.axis('off')

        for i in range(len(idx) * 2, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'Sample Predictions — TriSAFNet ({year})', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig, f'{output_dir}/sample_predictions_{year}')
        logger.info(f"  Saved sample_predictions_{year}")
    except Exception as e:
        logger.warning(f"  Could not generate sample predictions: {e}")


# ============================================================
# NEW PUBLICATION PLOTS
# ============================================================

def plot_architecture_diagram(input_shape, output_dir):
    """Generate TriSAFNet architecture diagram using tf.keras.utils.plot_model."""
    try:
        tf.keras.backend.clear_session()
        m = TriSAFNet(input_shape)
        tf.keras.utils.plot_model(
            m.model, to_file=f'{output_dir}/architecture_trisafnet.png',
            show_shapes=True, show_layer_names=True, dpi=300
        )
        logger.info("  Saved architecture_trisafnet.png")
    except Exception as e:
        logger.warning(f"  Could not generate architecture diagram: {e}")
    finally:
        try:
            del m
        except (NameError, UnboundLocalError):
            pass
        tf.keras.backend.clear_session()
        gc.collect()


def plot_benchmark_table(bench_df, year, output_dir):
    """Render benchmark CSV as a styled matplotlib table figure."""
    if bench_df.empty:
        return
    display_df = bench_df.copy()
    display_df['parameters'] = display_df['parameters'].apply(
        lambda x: f'{x:,}' if x > 0 else 'N/A')

    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(display_df) + 1)))
    ax.axis('off')
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D9E2F3')
    ax.set_title(f'Computational Benchmark — {year}', fontsize=12, fontweight='bold', pad=20)
    _save_fig(fig, f'{output_dir}/benchmark_table_{year}')


def plot_efficiency_radar(bench_df, results_df, year, output_dir):
    """Radar chart comparing all models on accuracy, params, train time, inference time."""
    if bench_df.empty or results_df.empty:
        return
    try:
        merged = results_df.merge(bench_df, on='model', how='inner')
        if merged.empty:
            return

        categories = ['Accuracy', 'Params (inv)', 'Train Speed', 'Inference Speed']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for _, row in merged.iterrows():
            acc_norm = row['accuracy']
            params_max = merged['parameters'].max() if merged['parameters'].max() > 0 else 1
            params_inv = 1 - (row['parameters'] / params_max) if params_max > 0 else 1
            train_max = merged['train_time_mean'].max() if merged['train_time_mean'].max() > 0 else 1
            train_speed = 1 - (row['train_time_mean'] / train_max)
            inf_max = merged['inference_time_mean'].max() if merged['inference_time_mean'].max() > 0 else 1
            inf_speed = 1 - (row['inference_time_mean'] / inf_max)
            values = [acc_norm, params_inv, train_speed, inf_speed]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=row['model'], markersize=4)
            ax.fill(angles, values, alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(f'Model Efficiency Comparison — {year}', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
        _save_fig(fig, f'{output_dir}/efficiency_radar_{year}')
    except Exception as e:
        logger.warning(f"  Could not generate radar chart: {e}")


def plot_per_class_precision_recall(all_predictions, year, output_dir):
    """Per-class precision/recall grouped bar chart for all models."""
    if not all_predictions:
        return
    model_names = list(all_predictions.keys())

    precs_all, recs_all = [], []
    for name in model_names:
        report = classification_report(all_predictions[name]['y_true'],
                                       all_predictions[name]['y_pred'],
                                       labels=_ACTIVE_CLASS_IDS, target_names=_ACTIVE_CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        precs_all.append([report[c]['precision'] for c in _ACTIVE_CLASS_NAMES])
        recs_all.append([report[c]['recall'] for c in _ACTIVE_CLASS_NAMES])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    prec_df = pd.DataFrame(precs_all, index=model_names, columns=_ACTIVE_CLASS_NAMES)
    rec_df = pd.DataFrame(recs_all, index=model_names, columns=_ACTIVE_CLASS_NAMES)

    prec_df.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'Per-Class Precision — {year}', fontweight='bold')
    ax1.set_ylabel('Precision')
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(fontsize=7, ncol=2)
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    rec_df.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title(f'Per-Class Recall — {year}', fontweight='bold')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0.5, 1.05)
    ax2.legend(fontsize=7, ncol=2)
    ax2.tick_params(axis='x', rotation=30, labelsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_fig(fig, f'{output_dir}/per_class_precision_recall_{year}')


def plot_training_convergence_comparison(all_histories, year, output_dir):
    """All DL models' loss curves overlaid on one plot for comparison."""
    models_with_history = {k: v for k, v in all_histories.items() if v is not None}
    if not models_with_history:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_history)))

    for (name, hist), color in zip(models_with_history.items(), colors):
        h = hist.history if hasattr(hist, 'history') else hist
        if 'val_loss' in h:
            ax1.plot(h['val_loss'], label=name, linewidth=1.2, color=color)
        if 'val_accuracy' in h:
            ax2.plot(h['val_accuracy'], label=name, linewidth=1.2, color=color)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title(f'Convergence Comparison — Val Loss ({year})', fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title(f'Convergence Comparison — Val Accuracy ({year})', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, f'{output_dir}/training_convergence_comparison_{year}')


def plot_class_distribution(y, year, output_dir):
    """Visualize class balance in the training data."""
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(unique)), counts,
                  color=[CLASS_COLORS[int(c)] if int(c) < len(CLASS_COLORS) else '#777777' for c in unique])
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_name_from_id(c) for c in unique], rotation=20, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Class Distribution — {year}', fontweight='bold')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    _save_fig(fig, f'{output_dir}/class_distribution_{year}')


def plot_hyperparameter_table(year, output_dir):
    """Render a table of all configuration values."""
    rows = []
    for key in sorted(dir(_cfg_module)):
        if key.startswith('_'):
            continue
        val = getattr(_cfg_module, key)
        if isinstance(val, (int, float, str, bool)):
            rows.append({'Parameter': key, 'Value': str(val)})
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.25)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title(f'Hyperparameter Configuration — {year}', fontsize=11,
                 fontweight='bold', pad=15)
    _save_fig(fig, f'{output_dir}/hyperparameter_table_{year}')


def plot_ablation_bar_chart(ablation_df, year, output_dir):
    """Grouped bar chart of ablation results."""
    if ablation_df is None or ablation_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ablation_df))
    w = 0.25

    ax.bar(x - w, ablation_df['mean_acc'], w, yerr=ablation_df['std_acc'],
           label='Accuracy', capsize=3)
    ax.bar(x, ablation_df['mean_kappa'], w, yerr=ablation_df['std_kappa'],
           label='Kappa', capsize=3)
    ax.bar(x + w, ablation_df['mean_f1'], w, yerr=ablation_df['std_f1'],
           label='F1 Score', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df['variant'], rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title(f'TriSAFNet Ablation Study — {year}', fontweight='bold')
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    _save_fig(fig, f'{output_dir}/ablation_bar_chart_{year}')


def plot_cross_year_heatmap(output_dir):
    """Render cross-year accuracy matrix as a heatmap."""
    csv_path = f"{output_dir}/cross_year_generalization.csv"
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return

    matrix = np.array([
        [df.iloc[0].get('accuracy', 0), df.iloc[0].get('accuracy', 0)],
        [df.iloc[1].get('accuracy', 0), df.iloc[1].get('accuracy', 0)]
    ])
    if len(df) >= 2:
        matrix = np.array([
            [1.0, df.iloc[0]['accuracy']],
            [df.iloc[1]['accuracy'], 1.0]
        ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=['2018', '2024'], yticklabels=['2018', '2024'], ax=ax)
    ax.set_xlabel('Test Year')
    ax.set_ylabel('Train Year')
    ax.set_title('Cross-Year Generalization — TriSAFNet', fontweight='bold')
    _save_fig(fig, f'{output_dir}/cross_year_heatmap')


# ============================================================
# CROSS-YEAR GENERALIZATION TEST
# ============================================================

def run_cross_year_test(output_dir):
    logger.info(f"\n{'='*50}")
    logger.info("CROSS-YEAR GENERALIZATION TEST")
    logger.info(f"{'='*50}")

    ps = _ACTIVE_PATCH_SIZE
    try:
        d18, m18 = load_npz_dataset(f"{DATA_DIR}/patches_2018_{ps}x{ps}.npz")
        d24, m24 = load_npz_dataset(f"{DATA_DIR}/patches_2024_{ps}x{ps}.npz")
    except FileNotFoundError as e:
        logger.info(f"  Skipping cross-year test: {e}")
        return

    if m18['n_bands'] != m24['n_bands']:
        logger.warning("  Skipping cross-year test due to band mismatch: "
                       f"2018={m18['n_bands']} vs 2024={m24['n_bands']}")
        return

    pairs = [
        ('Train 2018 -> Test 2024', d18['patches'], d18['labels'], d24['patches'], d24['labels']),
        ('Train 2024 -> Test 2018', d24['patches'], d24['labels'], d18['patches'], d18['labels']),
    ]

    rows = []
    for name, X_train, y_train, X_test, y_test in pairs:
        logger.info(f"  {name}")
        sync_active_classes(np.concatenate([y_train, y_test]))
        tf.keras.backend.clear_session()
        gc.collect()
        input_shape = X_train.shape[1:]
        model = TriSAFNet(input_shape)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        rows.append({
            'experiment': name,
            'accuracy': metrics['accuracy'],
            'kappa': metrics['kappa'],
            'f1_macro': metrics['f1_macro'],
        })
        logger.info(f"    Acc={metrics['accuracy']:.4f}, Kappa={metrics['kappa']:.4f}, "
                     f"F1={metrics['f1_macro']:.4f}")
        del model, y_pred
        tf.keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/cross_year_generalization.csv", index=False)
    logger.info("  Saved: cross_year_generalization.csv")

    try:
        with start_run(run_name="cross_year_test", nested=True):
            mlflow.set_tag('task', 'cross_year')
            for _, row in df.iterrows():
                safe = row['experiment'].replace(' ', '_').replace('->', 'to')
                log_metrics({
                    f'{safe}_acc': row['accuracy'],
                    f'{safe}_kappa': row['kappa'],
                    f'{safe}_f1': row['f1_macro'],
                })
            log_artifact_safe(f"{output_dir}/cross_year_generalization.csv")
    except Exception as e:
        logger.warning(f"  MLflow cross-year logging failed: {e}")


# ============================================================
# ABLATION STUDY
# ============================================================

def run_ablation_study(year, run_dir, logs_dir):
    """Run TriSAFNet ablation study with different component configurations."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  TriSAFNet ABLATION STUDY FOR YEAR: {year}")
    logger.info(f"{'='*70}")

    ps = _ACTIVE_PATCH_SIZE
    data_path = f"{DATA_DIR}/patches_{year}_{ps}x{ps}.npz"
    data, meta = load_npz_dataset(data_path)
    X = data['patches']
    y = data['labels']
    block_ids = data['block_ids']
    sync_active_classes(y)
    input_shape = X.shape[1:]
    logger.info(f"Detected schema: bands={meta['n_bands']}, patch={meta['patch_size']}, "
                f"classes={meta['class_ids']}")

    ablation_variants = [
        ('TriSAFNet (Full)',        dict(use_se=True,  use_cbam=True,  use_csa=True,  single_branch=False)),
        ('w/o CSA',               dict(use_se=True,  use_cbam=True,  use_csa=False, single_branch=False)),
        ('w/o CBAM',              dict(use_se=True,  use_cbam=False, use_csa=True,  single_branch=False)),
        ('w/o SE',                dict(use_se=False, use_cbam=True,  use_csa=True,  single_branch=False)),
        ('Single-Branch',         dict(use_se=True,  use_cbam=True,  use_csa=True,  single_branch=True)),
    ]

    noaug_path = f"{DATA_DIR}/patches_{year}_{ps}x{ps}_noaug.npz"
    noaug_arrays = None
    if Path(noaug_path).exists():
        _nf = np.load(noaug_path, allow_pickle=True)
        noaug_arrays = {
            'patches': np.array(_nf['patches']),
            'labels': np.array(_nf['labels']),
            'block_ids': np.array(_nf['block_ids']) if 'block_ids' in _nf.files else None,
        }
        _nf.close()
        ablation_variants.append(('w/o Physics Aug', dict(use_se=True, use_cbam=True,
                                                           use_csa=True, single_branch=False)))
    else:
        logger.info(f"  No noaug data found at {noaug_path}, skipping 'w/o Physics Aug' variant")

    if SPATIAL_CV:
        cv_generator = list(spatial_cv_splits(block_ids))
    else:
        cv_generator = list(standard_cv_splits(y))

    ablation_rows = []

    for variant_name, kwargs in ablation_variants:
        logger.info(f"\n  Ablation variant: {variant_name}")

        if variant_name == 'w/o Physics Aug' and noaug_arrays is not None:
            X_data = noaug_arrays['patches']
            y_data = noaug_arrays['labels']
            block_ids_data = noaug_arrays['block_ids']
            if SPATIAL_CV and block_ids_data is not None:
                cv_gen = list(spatial_cv_splits(block_ids_data))
            else:
                cv_gen = list(standard_cv_splits(y_data))
        else:
            X_data, y_data = X, y
            cv_gen = cv_generator

        fold_accs, fold_kappas, fold_f1s = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_gen):
            tf.keras.backend.clear_session()
            gc.collect()

            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            try:
                model = TriSAFNet(input_shape, **kwargs)
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)
                fold_accs.append(metrics['accuracy'])
                fold_kappas.append(metrics['kappa'])
                fold_f1s.append(metrics['f1_macro'])
                logger.info(f"    Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f}, "
                             f"Kappa={metrics['kappa']:.4f}, F1={metrics['f1_macro']:.4f}")
            except Exception as e:
                logger.error(f"    Fold {fold_idx+1} ERROR: {e}")
            # free GPU after every ablation fold
            try:
                del model
            except NameError:
                pass
            tf.keras.backend.clear_session()
            gc.collect()

        if fold_accs:
            ablation_rows.append({
                'variant': variant_name,
                'mean_acc': np.mean(fold_accs), 'std_acc': np.std(fold_accs),
                'mean_kappa': np.mean(fold_kappas), 'std_kappa': np.std(fold_kappas),
                'mean_f1': np.mean(fold_f1s), 'std_f1': np.std(fold_f1s),
            })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f"{run_dir}/ablation_results_{year}.csv", index=False)
    logger.info(f"\n  Ablation results saved to ablation_results_{year}.csv")
    logger.info(f"\n{ablation_df.to_string(index=False)}")

    plot_ablation_bar_chart(ablation_df, year, run_dir)

    try:
        with start_run(run_name=f"ablation_{year}", nested=True):
            mlflow.set_tag('task', 'ablation')
            mlflow.set_tag('year', str(year))
            for _, row in ablation_df.iterrows():
                safe = row['variant'].replace(' ', '_').replace('/', '_')
                log_metrics({
                    f'{safe}_acc': row['mean_acc'],
                    f'{safe}_kappa': row['mean_kappa'],
                    f'{safe}_f1': row['mean_f1'],
                })
            log_artifact_safe(f"{run_dir}/ablation_results_{year}.csv")
            log_artifact_safe(f"{run_dir}/ablation_bar_chart_{year}.png")
    except Exception as e:
        logger.warning(f"  MLflow ablation logging failed: {e}")

    return ablation_df


# ============================================================
# MAIN EXPERIMENT RUNNER
# ============================================================

def run_experiment(year, output_dir, logs_dir):
    """Run complete experiment for one year."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  RUNNING EXPERIMENTS FOR YEAR: {year}")
    logger.info(f"{'='*70}")

    ps = _ACTIVE_PATCH_SIZE
    data_path = f"{DATA_DIR}/patches_{year}_{ps}x{ps}.npz"
    logger.info(f"\nLoading data from {data_path}...")
    data, meta = load_npz_dataset(data_path)
    X = data['patches']
    y = data['labels']
    block_ids = data['block_ids']
    sync_active_classes(y)
    logger.info(f"Detected schema: bands={meta['n_bands']}, patch={meta['patch_size']}, "
                f"classes={meta['class_ids']}")

    input_shape = X.shape[1:]
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Input shape for models: {input_shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    plot_class_distribution(y, year, output_dir)

    # ---- DEFINE MODELS (9 models after trimming) ----
    model_configs = [
        ('TriSAFNet (Proposed)',    lambda: TriSAFNet(input_shape)),
        ('Lightweight CNN-RF',   lambda: LightweightCNNRF(input_shape)),
        ('CNN-LightGBM',         lambda: LightGBMCNNRF(input_shape)),
        ('CNN Only',             lambda: CNNOnly(input_shape)),
        ('MobileNetV2',          lambda: MobileNetV2Classifier(input_shape)),
        ('EfficientNet-B0',      lambda: EfficientNetB0Classifier(input_shape)),
        ('Compact ViT',          lambda: CompactViTClassifier(input_shape)),
        ('RF Only',              lambda: RFOnly()),
        ('XGBoost Only',         lambda: XGBoostOnly()),
    ]

    # ---- CV STRATEGY ----
    logger.info(f"\n{'='*50}")
    if SPATIAL_CV:
        logger.info(f"SPATIAL {N_FOLDS}-FOLD CROSS-VALIDATION")
        cv_generator = list(spatial_cv_splits(block_ids))
    else:
        logger.info(f"STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
        cv_generator = list(standard_cv_splits(y))
    logger.info(f"{'='*50}")

    all_results = []
    cv_details = {name: [] for name, _ in model_configs}
    all_predictions = {}
    all_histories = {}
    all_cms = {}
    bench_records = {name: [] for name, _ in model_configs}
    timing_rows = []

    # TriSAFNet per-fold tracking
    trisafnet_models_dir = Path(output_dir) / 'trisafnet_models'
    trisafnet_models_dir.mkdir(exist_ok=True, parents=True)
    best_trisafnet_acc = -1
    best_trisafnet_fold = -1

    # Models directory for all models
    models_dir = Path(output_dir) / 'models'
    models_dir.mkdir(exist_ok=True, parents=True)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_generator):
        logger.info(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        logger.info(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model_factory in model_configs:
            logger.info(f"\n  Training: {model_name}")
            tf.keras.backend.clear_session()
            gc.collect()

            try:
                tracemalloc.start()
                t0 = time.time()

                model = model_factory()
                model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

                train_time = time.time() - t0
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                t0_inf = time.time()
                y_pred = model.predict(X_test)
                inf_time = time.time() - t0_inf

                params = model.get_param_count() if hasattr(model, 'get_param_count') else 0

                history_obj = model.history if hasattr(model, 'history') else None
                best_val_loss = None
                best_val_acc = None
                epochs_completed = 0
                if history_obj is not None:
                    h = _history_dict(history_obj)
                    epochs_completed = len(h.get('loss', []))
                    if 'val_loss' in h and h['val_loss']:
                        best_val_loss = min(h['val_loss'])
                    if 'val_accuracy' in h and h['val_accuracy']:
                        best_val_acc = max(h['val_accuracy'])

                timing_row = {
                    'model': model_name, 'fold': fold_idx + 1,
                    'train_time_sec': round(train_time, 2),
                    'inference_time_sec': round(inf_time, 4),
                    'peak_memory_mb': round(peak_mem / 1024 / 1024, 1),
                    'parameters': params,
                    'epochs_completed': epochs_completed,
                    'best_val_loss': round(best_val_loss, 6) if best_val_loss is not None else None,
                    'best_val_acc': round(best_val_acc, 6) if best_val_acc is not None else None,
                }
                timing_rows.append(timing_row)

                bench_records[model_name].append({
                    'fold': fold_idx + 1,
                    'parameters': params,
                    'train_time_sec': round(train_time, 2),
                    'inference_time_sec': round(inf_time, 4),
                    'peak_memory_mb': round(peak_mem / 1024 / 1024, 1),
                })

                save_epoch_log(history_obj, model_name, fold_idx + 1, logs_dir)

                metrics = compute_metrics(y_test, y_pred)
                metrics['fold'] = fold_idx + 1
                metrics['model'] = model_name
                cv_details[model_name].append(metrics)

                if model_name == 'TriSAFNet (Proposed)' and hasattr(model, 'model'):
                    fold_model_path = trisafnet_models_dir / f'fold_{fold_idx+1}_model.keras'
                    model.model.save(str(fold_model_path))
                    with open(trisafnet_models_dir / f'fold_{fold_idx+1}_meta.json', 'w') as mf:
                        json.dump({'fold': fold_idx+1,
                                   'accuracy': float(metrics['accuracy']),
                                   'kappa': float(metrics['kappa']),
                                   'f1_macro': float(metrics['f1_macro'])}, mf, indent=2)
                    if metrics['accuracy'] > best_trisafnet_acc:
                        best_trisafnet_acc = metrics['accuracy']
                        best_trisafnet_fold = fold_idx + 1

                if fold_idx == N_FOLDS - 1:
                    all_predictions[model_name] = {'y_true': y_test, 'y_pred': y_pred}
                    all_cms[model_name] = metrics['confusion_matrix']

                    if hasattr(model, 'history') and model.history is not None:
                        all_histories[model_name] = model.history
                    else:
                        all_histories[model_name] = None

                    plot_confusion_matrix(
                        metrics['confusion_matrix'], _ACTIVE_CLASS_NAMES,
                        f'{model_name} — {year} (Fold {fold_idx+1})',
                        f'{output_dir}/cm_{model_name.replace(" ", "_")}_{year}.png'
                    )

                    if hasattr(model, 'history') and model.history:
                        plot_training_curves(
                            model.history, model_name,
                            f'{output_dir}/training_curve_{model_name.replace(" ", "_")}_{year}.png'
                        )

                    if model_name == 'Lightweight CNN-RF' and hasattr(model, 'get_feature_importances'):
                        plot_feature_importance(
                            model.get_feature_importances(), model_name,
                            f'{output_dir}/feature_importance_{year}.png'
                        )

                    if model_name == 'TriSAFNet (Proposed)' and hasattr(model, 'predict_proba'):
                        plot_sample_predictions(model, X_test, y_test, year, output_dir)

                    save_model_artifact(model, model_name, str(models_dir))

                logger.info(f"    Acc={metrics['accuracy']:.4f}, Kappa={metrics['kappa']:.4f}, "
                             f"F1={metrics['f1_macro']:.4f} | Train={train_time:.1f}s "
                             f"Inf={inf_time:.3f}s Mem={peak_mem/1024/1024:.1f}MB")

            except Exception as e:
                logger.error(f"    ERROR: {e}")
                traceback.print_exc()
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
            finally:
                try:
                    del model
                except NameError:
                    pass
                tf.keras.backend.clear_session()
                gc.collect()

    # ---- SAVE MODEL TIMINGS LOG ----
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(str(Path(logs_dir) / 'model_timings.csv'), index=False)
    logger.info(f"  Saved model_timings.csv to logs/")

    # ---- COPY BEST TriSAFNet MODEL ----
    if best_trisafnet_fold > 0:
        src = trisafnet_models_dir / f'fold_{best_trisafnet_fold}_model.keras'
        dst = trisafnet_models_dir / 'best_model.keras'
        shutil.copy2(str(src), str(dst))
        with open(trisafnet_models_dir / 'best_model.json', 'w') as bf:
            json.dump({'best_fold': best_trisafnet_fold,
                       'accuracy': float(best_trisafnet_acc),
                       'year': year}, bf, indent=2)
        logger.info(f"\n  Best TriSAFNet: fold {best_trisafnet_fold} (acc={best_trisafnet_acc:.4f})")
        logger.info(f"  Saved to: {dst}")

    # ---- AGGREGATE CV RESULTS ----
    logger.info(f"\n{'='*50}")
    logger.info(f"AGGREGATED CROSS-VALIDATION RESULTS - {year}")
    logger.info(f"{'='*50}")

    summary_rows = []
    for model_name, folds in cv_details.items():
        if not folds:
            continue
        accs = [f['accuracy'] for f in folds]
        kappas = [f['kappa'] for f in folds]
        f1s = [f['f1_macro'] for f in folds]
        precs = [f['precision_macro'] for f in folds]
        recs = [f['recall_macro'] for f in folds]
        row = {
            'model': model_name,
            'accuracy': np.mean(accs), 'accuracy_std': np.std(accs),
            'kappa': np.mean(kappas), 'kappa_std': np.std(kappas),
            'f1_macro': np.mean(f1s), 'f1_std': np.std(f1s),
            'precision_macro': np.mean(precs), 'recall_macro': np.mean(recs),
        }
        summary_rows.append(row)
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy: {row['accuracy']:.4f} +/- {row['accuracy_std']:.4f}")
        logger.info(f"  Kappa:    {row['kappa']:.4f} +/- {row['kappa_std']:.4f}")
        logger.info(f"  F1 Score: {row['f1_macro']:.4f} +/- {row['f1_std']:.4f}")

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(f"{output_dir}/results_{year}.csv", index=False)

    # ---- PER-FOLD RESULTS TABLE ----
    fold_rows = []
    for model_name, folds in cv_details.items():
        for f in folds:
            fold_rows.append({
                'model': model_name, 'fold': f['fold'],
                'accuracy': f['accuracy'], 'kappa': f['kappa'], 'f1_score': f['f1_macro']
            })
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f"{output_dir}/cv_folds_{year}.csv", index=False)

    # ---- COMPREHENSIVE PLOTS (parallel, except TF-dependent ones) ----
    logger.info(f"\nGenerating all plots in parallel...")

    proposed_name = 'TriSAFNet (Proposed)'
    plot_jobs = [
        ("model_comparison",          lambda: plot_comparison_chart(results_df, year, f"{output_dir}/model_comparison_{year}.png")),
        ("all_confusion_matrices",    lambda: plot_all_confusion_matrices(all_cms, year, output_dir)),
        ("all_training_curves",       lambda: plot_all_training_curves(all_histories, year, output_dir)),
        ("per_class_f1_heatmap",      lambda: plot_per_class_all_models(all_predictions, year, output_dir)),
        ("per_class_precision_recall", lambda: plot_per_class_precision_recall(all_predictions, year, output_dir)),
        ("convergence_comparison",    lambda: plot_training_convergence_comparison(all_histories, year, output_dir)),
    ]
    if proposed_name in all_predictions and proposed_name in all_cms:
        plot_jobs.append(("proposed_detail", lambda: plot_proposed_model_detail(
            all_histories.get(proposed_name), all_cms[proposed_name],
            all_predictions[proposed_name]['y_true'],
            all_predictions[proposed_name]['y_pred'], year, output_dir)))

    with ThreadPoolExecutor(max_workers=min(len(plot_jobs), os.cpu_count() or 4)) as pool:
        futs = {pool.submit(fn): name for name, fn in plot_jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
                logger.info(f"  Plot OK: {name}")
            except Exception as e:
                logger.warning(f"  Plot FAILED: {name} — {e}")

    # Architecture diagram uses TF clear_session — must run on main thread
    plot_architecture_diagram(input_shape, output_dir)

    # ---- MCNEMAR TESTS ----
    logger.info(f"\n{'='*50}")
    logger.info(f"McNEMAR'S TEST RESULTS - {year}")
    logger.info(f"{'='*50}")

    if proposed_name in all_predictions:
        y_true = all_predictions[proposed_name]['y_true']
        y_pred_proposed = all_predictions[proposed_name]['y_pred']
        mcnemar_rows = []
        for model_name, preds in all_predictions.items():
            if model_name == proposed_name:
                continue
            chi2_stat, p_val = mcnemar_test(y_true, y_pred_proposed, preds['y_pred'])
            sig = "YES" if p_val < 0.05 else "NO"
            mcnemar_rows.append({
                'comparison': f'{proposed_name} vs {model_name}',
                'chi2': chi2_stat, 'p_value': p_val, 'significant_at_0.05': sig
            })
            logger.info(f"  vs {model_name}: chi2={chi2_stat:.3f}, p={p_val:.4f} ({sig})")
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        mcnemar_df.to_csv(f"{output_dir}/mcnemar_{year}.csv", index=False)

    # ---- FRIEDMAN + NEMENYI TEST ----
    logger.info(f"\n{'='*50}")
    logger.info(f"FRIEDMAN + NEMENYI TEST - {year}")
    logger.info(f"{'='*50}")

    try:
        model_names_for_test = [n for n in cv_details.keys() if len(cv_details[n]) == N_FOLDS]
        if len(model_names_for_test) >= 3:
            acc_matrix = np.array([
                [f['accuracy'] for f in cv_details[m]]
                for m in model_names_for_test
            ]).T

            stat, p_val = friedmanchisquare(*[acc_matrix[:, i] for i in range(acc_matrix.shape[1])])
            logger.info(f"  Friedman chi2={stat:.3f}, p={p_val:.6f}")

            if p_val < 0.05:
                logger.info("  Significant differences found. Running Nemenyi post-hoc...")
                try:
                    import scikit_posthocs as sp
                    nemenyi = sp.posthoc_nemenyi_friedman(acc_matrix)
                    nemenyi.index = model_names_for_test
                    nemenyi.columns = model_names_for_test
                    nemenyi.to_csv(f"{output_dir}/nemenyi_{year}.csv")
                    logger.info(f"  Nemenyi results saved to nemenyi_{year}.csv")
                except ImportError:
                    logger.info("  scikit-posthocs not installed, skipping Nemenyi test")
            else:
                logger.info("  No significant differences (p >= 0.05), skipping post-hoc.")
        else:
            logger.info(f"  Need >= 3 models with {N_FOLDS} folds each for Friedman test.")
    except Exception as e:
        logger.error(f"  Friedman test error: {e}")

    # ---- RF SENSITIVITY ----
    rf_sens_df = rf_sensitivity_analysis(
        X[cv_generator[-1][0]], y[cv_generator[-1][0]],
        X[cv_generator[-1][1]], y[cv_generator[-1][1]],
        input_shape, output_dir
    )
    rf_sens_df.to_csv(f"{output_dir}/rf_sensitivity_{year}.csv", index=False)

    # ---- COMPUTATIONAL BENCHMARK ----
    logger.info(f"\n{'='*50}")
    logger.info(f"COMPUTATIONAL BENCHMARK - {year}")
    logger.info(f"{'='*50}")

    bench_summary = []
    for model_name, _ in model_configs:
        recs = bench_records.get(model_name, [])
        if not recs:
            continue
        bench_summary.append({
            'model': model_name,
            'parameters': recs[0]['parameters'],
            'train_time_mean': round(np.mean([r['train_time_sec'] for r in recs]), 2),
            'train_time_std': round(np.std([r['train_time_sec'] for r in recs]), 2),
            'inference_time_mean': round(np.mean([r['inference_time_sec'] for r in recs]), 4),
            'peak_memory_mb': round(np.mean([r['peak_memory_mb'] for r in recs]), 1),
        })
        logger.info(f"  {model_name}: params={recs[0]['parameters']}, "
                     f"train={bench_summary[-1]['train_time_mean']:.2f}s, "
                     f"mem={bench_summary[-1]['peak_memory_mb']:.1f}MB")

    bench_df = pd.DataFrame(bench_summary)
    bench_df.to_csv(f"{output_dir}/benchmark_{year}.csv", index=False)

    # ---- PER-CLASS METRICS TABLE ----
    if proposed_name in all_predictions:
        y_true = all_predictions[proposed_name]['y_true']
        y_pred = all_predictions[proposed_name]['y_pred']
        per_class_rows = []
        report = classification_report(y_true, y_pred, labels=_ACTIVE_CLASS_IDS, target_names=_ACTIVE_CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        for cls_name in _ACTIVE_CLASS_NAMES:
            per_class_rows.append({
                'class': cls_name,
                'precision': report[cls_name]['precision'],
                'recall': report[cls_name]['recall'],
                'f1_score': report[cls_name]['f1-score'],
                'support': report[cls_name]['support']
            })
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_df.to_csv(f"{output_dir}/per_class_metrics_{year}.csv", index=False)
        logger.info(f"\nPer-class metrics (Proposed Model):")
        logger.info(f"\n{per_class_df.to_string(index=False)}")

    # ---- REMAINING PLOTS (parallel) ----
    late_plots = [
        ("benchmark_table",     lambda: plot_benchmark_table(bench_df, year, output_dir)),
        ("efficiency_radar",    lambda: plot_efficiency_radar(bench_df, results_df, year, output_dir)),
        ("hyperparameter_table", lambda: plot_hyperparameter_table(year, output_dir)),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(fn): name for name, fn in late_plots}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"  Late plot FAILED: {futs[fut]} — {e}")

    # ---- MLFLOW: LOG PER-MODEL NESTED RUNS + ALL ARTIFACTS ----
    try:
        for _, row in results_df.iterrows():
            model_name = row['model']
            with start_run(run_name=f"{model_name}_{year}", nested=True):
                log_metrics({
                    'accuracy': row['accuracy'],
                    'accuracy_std': row['accuracy_std'],
                    'kappa': row['kappa'],
                    'kappa_std': row['kappa_std'],
                    'f1_macro': row['f1_macro'],
                    'f1_std': row['f1_std'],
                    'precision_macro': row['precision_macro'],
                    'recall_macro': row['recall_macro'],
                })
                mlflow.set_tag('model', model_name)
                mlflow.set_tag('year', str(year))

                bench_row = bench_df[bench_df['model'] == model_name]
                if not bench_row.empty:
                    br = bench_row.iloc[0]
                    log_metrics({
                        'parameters': br['parameters'],
                        'train_time_mean': br['train_time_mean'],
                        'inference_time_mean': br['inference_time_mean'],
                        'peak_memory_mb': br['peak_memory_mb'],
                    })

                # Per-class P/R/F1 for every model
                if model_name in all_predictions:
                    _yt = all_predictions[model_name]['y_true']
                    _yp = all_predictions[model_name]['y_pred']
                    _rpt = classification_report(
                        _yt, _yp, labels=_ACTIVE_CLASS_IDS,
                        target_names=_ACTIVE_CLASS_NAMES,
                        output_dict=True, zero_division=0)
                    for cls_name in _ACTIVE_CLASS_NAMES:
                        safe_cls = cls_name.replace(' ', '_')
                        log_metrics({
                            f'{safe_cls}_precision': _rpt[cls_name]['precision'],
                            f'{safe_cls}_recall': _rpt[cls_name]['recall'],
                            f'{safe_cls}_f1': _rpt[cls_name]['f1-score'],
                        })

                # Epoch-level training curves as step metrics
                hist = all_histories.get(model_name)
                h = _history_dict(hist)
                if h is not None:
                    _epoch_keys = [('loss', 'train_loss'), ('accuracy', 'train_accuracy'),
                                   ('val_loss', 'val_loss'), ('val_accuracy', 'val_accuracy')]
                    n_epochs = max((len(h.get(k, [])) for k, _ in _epoch_keys), default=0)
                    for epoch_i in range(n_epochs):
                        step_metrics = {}
                        for src_key, log_key in _epoch_keys:
                            vals = h.get(src_key, [])
                            if epoch_i < len(vals):
                                step_metrics[log_key] = vals[epoch_i]
                        if step_metrics:
                            log_metrics(step_metrics, step=epoch_i)

        log_artifacts_dir(output_dir, artifact_path=f"experiment_{year}")
        logger.info(f"  MLflow: logged nested runs and artifacts for {year}")
    except Exception as e:
        logger.warning(f"  MLflow logging (run_experiment) failed: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"ALL OUTPUTS SAVED TO: {output_dir}/")
    logger.info(f"{'='*50}")

    return results_df


# ============================================================
# RUN CONFIG SNAPSHOT
# ============================================================

def save_run_config(run_dir, year):
    config_snapshot = {}
    for key in sorted(dir(_cfg_module)):
        if key.startswith('_'):
            continue
        val = getattr(_cfg_module, key)
        if isinstance(val, (int, float, str, bool, list)):
            config_snapshot[key] = val
    config_snapshot['_run_year'] = year
    config_snapshot['_run_timestamp'] = datetime.now().isoformat()
    out_path = Path(run_dir) / 'run_config.json'
    with open(out_path, 'w') as f:
        json.dump(config_snapshot, f, indent=2)
    logger.info(f"  Config snapshot saved to: {out_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CNN-RF experiments')
    parser.add_argument('--year', type=str, default='both',
                        choices=['2018', '2024', 'both'])
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE,
                        help=f'Patch size to use (default: {PATCH_SIZE}). '
                             f'Must match the patches file from 02_data_pipeline.py.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output files (e.g., _ps9 for patch size 9)')
    parser.add_argument('--ablation', action='store_true',
                        help='Run patch size ablation')
    parser.add_argument('--ablation-trisafnet', action='store_true',
                        help='Run TriSAFNet component ablation study')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow tracking')
    args = parser.parse_args()

    if not args.suffix and args.patch_size != PATCH_SIZE:
        args.suffix = f'_ps{args.patch_size}'

    _ACTIVE_PATCH_SIZE = args.patch_size

    setup_plot_style()

    output_dir = f"{OUTPUT_DIR}{args.suffix}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # ---- MLFLOW INIT ----
    if not args.no_mlflow:
        try:
            init_mlflow(experiment_name="LULC_Experiments")
        except Exception as e:
            print(f"  MLflow init failed ({e}); continuing without MLflow.")
            args.no_mlflow = True

    years = [2018, 2024] if args.year == 'both' else [int(args.year)]

    for year in years:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        run_name = f"{timestamp}_{year}{args.suffix}"
        run_dir = Path(RUNS_DIR) / run_name
        run_dir.mkdir(exist_ok=True, parents=True)

        logs_dir = setup_logging(str(run_dir))

        logger.info(f"\nRun directory: {run_dir}")

        # Start MLflow parent run for this year
        _mlflow_run = None
        if not args.no_mlflow:
            try:
                _mlflow_run = start_run(
                    run_name=f"experiment_{year}{args.suffix}",
                    tags={'year': str(year), 'patch_size': str(args.patch_size)},
                )
                log_config(_cfg_module)
            except Exception as e:
                logger.warning(f"  MLflow parent run failed: {e}")

        try:
            save_run_config(str(run_dir), year)
            results = run_experiment(year, str(run_dir), str(logs_dir))

            if args.ablation_trisafnet:
                run_ablation_study(year, str(run_dir), str(logs_dir))

            # ---- BUILD ARTIFACT MANIFEST ----
            expected = {}
            for suffix_ext in ['csv', 'png', 'pdf']:
                for fp in run_dir.rglob(f'*.{suffix_ext}'):
                    expected[fp.stem] = str(fp)
            manifest_path = build_artifact_manifest(str(run_dir), expected)

            if not args.no_mlflow:
                try:
                    log_artifact_safe(str(manifest_path))
                except Exception:
                    pass

            # ---- MIRROR TO LATEST OUTPUTS DIR ----
            for f in run_dir.glob('*'):
                if f.is_file():
                    dest = Path(output_dir) / f.name
                    shutil.copy2(str(f), str(dest))
            for sub in run_dir.iterdir():
                if sub.is_dir():
                    dest_sub = Path(output_dir) / sub.name
                    if dest_sub.exists():
                        shutil.rmtree(str(dest_sub))
                    shutil.copytree(str(sub), str(dest_sub))
            logger.info(f"\n  Results also copied to: {output_dir}/")

        except FileNotFoundError as e:
            logger.error(f"\nERROR: {e}")
            logger.error("Make sure you've run the data pipeline first:")
            logger.error("  1. Run 01_gee_export.js in GEE Code Editor")
            logger.error("  2. Download exported files to ./data/")
            logger.error("  3. Run: python scripts/02_data_pipeline.py")
            logger.error("  4. Then re-run this script")
        finally:
            if _mlflow_run is not None:
                try:
                    end_run()
                except Exception:
                    pass

    if args.year == 'both':
        _mlflow_cross = None
        if not args.no_mlflow:
            try:
                _mlflow_cross = start_run(
                    run_name="cross_year_generalization",
                    tags={'task': 'cross_year'},
                )
            except Exception:
                pass

        run_cross_year_test(output_dir)
        plot_cross_year_heatmap(output_dir)

        if _mlflow_cross is not None:
            try:
                log_artifact_safe(f"{output_dir}/cross_year_generalization.csv")
                log_artifact_safe(f"{output_dir}/cross_year_heatmap.png")
                end_run()
            except Exception:
                pass

    logger.info("\n" + "=" * 70)
    logger.info("  ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nTimestamped run(s) saved in: {RUNS_DIR}/")
    logger.info(f"Latest outputs also in:      {output_dir}/")
    logger.info(f"MLflow tracking URI:         file:./mlruns")
    logger.info("\nFiles generated for the paper:")
    logger.info("  - run_config.json           (Config snapshot)")
    logger.info("  - artifact_manifest.json    (Artifact status)")
    logger.info("  - results_YYYY.csv          (Model comparison)")
    logger.info("  - cv_folds_YYYY.csv         (Per-fold CV results)")
    logger.info("  - per_class_metrics_YYYY.csv (Per-class P/R/F1)")
    logger.info("  - benchmark_YYYY.csv        (Computational benchmark)")
    logger.info("  - benchmark_table_YYYY.png  (Styled benchmark table)")
    logger.info("  - efficiency_radar_YYYY.png (Efficiency radar chart)")
    logger.info("  - mcnemar_YYYY.csv          (Statistical significance)")
    logger.info("  - nemenyi_YYYY.csv          (Post-hoc tests)")
    logger.info("  - ablation_results_YYYY.csv (Ablation study)")
    logger.info("  - ablation_bar_chart_YYYY.png")
    logger.info("  - rf_sensitivity_YYYY.csv   (RF tree count sensitivity)")
    logger.info("  - cross_year_generalization.csv / cross_year_heatmap.png")
    logger.info("  - class_distribution_YYYY.png")
    logger.info("  - architecture_trisafnet.png  (Architecture diagram)")
    logger.info("  - hyperparameter_table_YYYY.png")
    logger.info("  - per_class_precision_recall_YYYY.png")
    logger.info("  - training_convergence_comparison_YYYY.png")
    logger.info("  - sample_predictions_YYYY.png")
    logger.info("  - trisafnet_models/           (TriSAFNet per fold + best)")
    logger.info("  - models/                   (All trained model artifacts)")
    logger.info("  - logs/                     (Structured experiment logs)")
    logger.info("\nMLflow UI: run 'mlflow ui' to browse experiments.")
