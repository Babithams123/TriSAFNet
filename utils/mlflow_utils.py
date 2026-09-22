"""
utils/mlflow_utils.py
Thin helpers for MLflow experiment tracking in the LULC pipeline.

Keeps MLflow wiring out of the main scripts so they stay readable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import mlflow

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

def init_mlflow(experiment_name="LULC_Pipeline", tracking_uri=None):
    """Set up the MLflow tracking URI and experiment.

    Returns the Experiment object (or None when MLflow is unavailable).
    """
    if tracking_uri is None:
        tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)


# ------------------------------------------------------------------
# Run helpers
# ------------------------------------------------------------------

def start_run(run_name, tags=None, nested=False):
    """Start an MLflow run (parent or nested)."""
    return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)


def end_run():
    """End the active MLflow run."""
    mlflow.end_run()


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------

def log_config(config_module):
    """Log every scalar config value as an MLflow parameter."""
    for key in sorted(dir(config_module)):
        if key.startswith("_"):
            continue
        val = getattr(config_module, key)
        if isinstance(val, (int, float, str, bool)):
            try:
                mlflow.log_param(key, val)
            except mlflow.exceptions.MlflowException:
                pass
        elif isinstance(val, list) and len(str(val)) < 500:
            try:
                mlflow.log_param(key, str(val))
            except mlflow.exceptions.MlflowException:
                pass


def log_metrics(metrics_dict, step=None, prefix=""):
    """Log a flat dict of numeric metrics (accepts numpy scalars)."""
    for key, val in metrics_dict.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            name = f"{prefix}{key}" if prefix else key
            try:
                mlflow.log_metric(name, float(val), step=step)
            except Exception:
                pass


def log_artifact_safe(path):
    """Log a single file as an artifact (no-op if the file is missing)."""
    if Path(path).exists():
        mlflow.log_artifact(str(path))


def log_artifacts_dir(dir_path, artifact_path=None):
    """Log every file in *dir_path* as MLflow artifacts."""
    dp = Path(dir_path)
    if dp.exists() and dp.is_dir():
        mlflow.log_artifacts(str(dp), artifact_path=artifact_path)


def log_figure(fig, artifact_name):
    """Log a matplotlib Figure directly as an image artifact."""
    try:
        mlflow.log_figure(fig, artifact_name)
    except Exception:
        pass


def log_dict_artifact(data, artifact_name):
    """Serialise *data* as JSON and log it as an artifact."""
    try:
        mlflow.log_dict(data, artifact_name)
    except Exception:
        pass


# ------------------------------------------------------------------
# Artifact manifest
# ------------------------------------------------------------------

def build_artifact_manifest(run_dir, expected_artifacts):
    """Create an ``artifact_manifest.json`` recording the status of each
    expected output file.

    Parameters
    ----------
    run_dir : str | Path
        Root directory of the current experiment run.
    expected_artifacts : dict[str, str | Path]
        Mapping of *logical name* -> *file path* for every artifact that the
        pipeline is supposed to produce.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "artifacts": {},
    }
    for name, path in expected_artifacts.items():
        p = Path(path)
        exists = p.exists()
        manifest["artifacts"][name] = {
            "path": str(path),
            "status": "created" if exists else "missing",
            "size_bytes": p.stat().st_size if exists else 0,
        }

    manifest_path = Path(run_dir) / "artifact_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"  Artifact manifest written to {manifest_path}")
    return manifest_path
