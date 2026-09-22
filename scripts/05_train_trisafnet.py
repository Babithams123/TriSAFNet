"""
05_train_trisafnet.py
Train TriSAFNet only with cross-validation, save models for every fold,
and select the best model based on validation accuracy.

Usage:
    python scripts/05_train_trisafnet.py --year 2018
    python scripts/05_train_trisafnet.py --year 2024
    python scripts/05_train_trisafnet.py --year both

Outputs (in ./outputs/trisafnet_models/):
    - fold_{i}_model.keras         (saved model for each CV fold)
    - best_model.keras             (copy of the best fold's model)
    - best_model.json              (manifest with best fold info & metrics)
    - trisafnet_cv_results_{year}.csv
    - trisafnet_training_curves_{year}.png
    - trisafnet_confusion_matrices_{year}.png
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
import json
import shutil
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import *
from models import TriSAFNet

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Compute all classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    prec_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'kappa': kappa,
        'f1_macro': f1_macro,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'per_class': report,
        'confusion_matrix': cm
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def spatial_cv_splits(block_ids, n_folds=N_FOLDS):
    """Generate train/test indices using spatial blocks."""
    unique_blocks = np.unique(block_ids)
    for test_block in unique_blocks[:n_folds]:
        test_idx = np.where(block_ids == test_block)[0]
        train_idx = np.where(block_ids != test_block)[0]
        yield train_idx, test_idx


def standard_cv_splits(labels, n_folds=N_FOLDS):
    """Standard stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        yield train_idx, test_idx


# ============================================================
# PLOTTING
# ============================================================

def plot_training_curves_all_folds(all_histories, year, output_path):
    """Plot training/val loss & accuracy for all folds as subplots."""
    n = len(all_histories)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx, (fold_name, hist) in enumerate(all_histories.items()):
        h = hist.history if hasattr(hist, 'history') else hist

        # Loss
        ax = axes[idx, 0]
        ax.plot(h['loss'], label='Train Loss', linewidth=1.5, color='#2196F3')
        if 'val_loss' in h:
            ax.plot(h['val_loss'], label='Val Loss', linewidth=1.5, color='#F44336')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{fold_name} — Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[idx, 1]
        ax.plot(h['accuracy'], label='Train Acc', linewidth=1.5, color='#4CAF50')
        if 'val_accuracy' in h:
            ax.plot(h['val_accuracy'], label='Val Acc', linewidth=1.5, color='#FF9800')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{fold_name} — Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'TriSAFNet Training Curves — All Folds ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices_all_folds(all_cms, year, output_path):
    """Plot confusion matrices for all folds as subplots."""
    n = len(all_cms)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (fold_name, cm) in enumerate(all_cms.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(fold_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'TriSAFNet Confusion Matrices — All Folds ({year})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# MAIN TRAINING
# ============================================================

def train_trisafnet(year, model_dir):
    """Train TriSAFNet with CV, save all fold models, select best."""
    print(f"\n{'='*70}")
    print(f"  TRAINING TriSAFNet FOR YEAR: {year}")
    print(f"{'='*70}")

    # Load data
    data_path = f"{DATA_DIR}/patches_{year}_{PATCH_SIZE}x{PATCH_SIZE}.npz"
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data['patches']
    y = data['labels']
    block_ids = data['block_ids']

    input_shape = X.shape[1:]
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # CV strategy
    print(f"\n{'='*50}")
    if SPATIAL_CV:
        print(f"SPATIAL {N_FOLDS}-FOLD CROSS-VALIDATION")
        cv_generator = list(spatial_cv_splits(block_ids))
    else:
        print(f"STRATIFIED {N_FOLDS}-FOLD CROSS-VALIDATION")
        cv_generator = list(standard_cv_splits(y))
    print(f"{'='*50}")

    fold_results = []
    all_histories = {}
    all_cms = {}
    best_acc = -1
    best_fold = -1

    for fold_idx, (train_idx, test_idx) in enumerate(cv_generator):
        fold_name = f"Fold {fold_idx + 1}"
        print(f"\n--- {fold_name}/{N_FOLDS} ---")
        print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tf.keras.backend.clear_session()

        try:
            model = TriSAFNet(input_shape)
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            metrics['fold'] = fold_idx + 1

            # Save model for this fold
            model_path = Path(model_dir) / f"fold_{fold_idx + 1}_model_{year}.keras"
            model.model.save(str(model_path))
            print(f"  Model saved: {model_path}")

            # Track metrics
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': metrics['accuracy'],
                'kappa': metrics['kappa'],
                'f1_macro': metrics['f1_macro'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'model_path': str(model_path)
            })

            # Store history & confusion matrix
            if hasattr(model, 'history') and model.history is not None:
                all_histories[fold_name] = model.history
            all_cms[fold_name] = metrics['confusion_matrix']

            print(f"  Acc={metrics['accuracy']:.4f}, Kappa={metrics['kappa']:.4f}, "
                  f"F1={metrics['f1_macro']:.4f}")

            # Track best
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_fold = fold_idx + 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- RESULTS SUMMARY ----
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS — TriSAFNet ({year})")
    print(f"{'='*50}")

    results_df = pd.DataFrame(fold_results)

    for _, row in results_df.iterrows():
        marker = " *** BEST ***" if row['fold'] == best_fold else ""
        print(f"  Fold {int(row['fold'])}: Acc={row['accuracy']:.4f}, "
              f"Kappa={row['kappa']:.4f}, F1={row['f1_macro']:.4f}{marker}")

    print(f"\n  Mean Accuracy:  {results_df['accuracy'].mean():.4f} "
          f"+/- {results_df['accuracy'].std():.4f}")
    print(f"  Mean Kappa:     {results_df['kappa'].mean():.4f} "
          f"+/- {results_df['kappa'].std():.4f}")
    print(f"  Mean F1:        {results_df['f1_macro'].mean():.4f} "
          f"+/- {results_df['f1_macro'].std():.4f}")
    print(f"\n  Best fold: {best_fold} (accuracy={best_acc:.4f})")

    # Save CSV
    csv_path = Path(model_dir) / f"trisafnet_cv_results_{year}.csv"
    results_df.to_csv(str(csv_path), index=False)
    print(f"  Results saved: {csv_path}")

    # ---- COPY BEST MODEL ----
    best_model_src = Path(model_dir) / f"fold_{best_fold}_model_{year}.keras"
    best_model_dst = Path(model_dir) / f"best_model_{year}.keras"
    shutil.copy2(str(best_model_src), str(best_model_dst))
    print(f"  Best model copied to: {best_model_dst}")

    # ---- BEST MODEL MANIFEST ----
    best_row = results_df[results_df['fold'] == best_fold].iloc[0]
    manifest = {
        'year': int(year),
        'best_fold': int(best_fold),
        'accuracy': float(best_row['accuracy']),
        'kappa': float(best_row['kappa']),
        'f1_macro': float(best_row['f1_macro']),
        'precision_macro': float(best_row['precision_macro']),
        'recall_macro': float(best_row['recall_macro']),
        'model_path': str(best_model_dst),
        'source_fold_path': str(best_model_src),
        'n_folds': N_FOLDS,
        'cv_type': 'spatial' if SPATIAL_CV else 'stratified',
        'timestamp': datetime.now().isoformat(),
        'all_fold_accuracies': results_df['accuracy'].tolist(),
    }
    manifest_path = Path(model_dir) / f"best_model_{year}.json"
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved: {manifest_path}")

    # ---- PLOTS ----
    print(f"\nGenerating plots...")
    plot_training_curves_all_folds(
        all_histories, year,
        str(Path(model_dir) / f"trisafnet_training_curves_{year}.png")
    )
    plot_confusion_matrices_all_folds(
        all_cms, year,
        str(Path(model_dir) / f"trisafnet_confusion_matrices_{year}.png")
    )
    print(f"  Plots saved.")

    return results_df, best_fold


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TriSAFNet with CV and save best model')
    parser.add_argument('--year', type=str, default='both',
                        choices=['2018', '2024', 'both'])
    args = parser.parse_args()

    model_dir = Path(OUTPUT_DIR) / "trisafnet_models"
    model_dir.mkdir(exist_ok=True, parents=True)

    years = [2018, 2024] if args.year == 'both' else [int(args.year)]

    for year in years:
        results_df, best_fold = train_trisafnet(year, str(model_dir))

    print("\n" + "=" * 70)
    print("  TriSAFNet TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nSaved models and results in: {model_dir}/")
    print(f"  - fold_*_model_*.keras   (all fold models)")
    print(f"  - best_model_*.keras     (best model per year)")
    print(f"  - best_model_*.json      (manifest with metrics)")
    print(f"  - trisafnet_cv_results_*.csv")
    print(f"  - trisafnet_training_curves_*.png")
    print(f"  - trisafnet_confusion_matrices_*.png")
    print(f"\nNext step: python scripts/06_generate_maps_trisafnet.py --year both")
