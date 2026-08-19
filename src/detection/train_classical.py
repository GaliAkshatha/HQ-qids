"""
src/detection/train_classical.py

Trains Random Forest + XGBoost + Isolation Forest for the classical IDS,
persists them plus preprocessing artifacts, evaluates once against the
held-out test split, and persists metrics.

Explicitly NOT doing: hyperparameter search against the test set. Model
settings below are reasonable, fixed defaults (documented inline), chosen
once, not tuned against test-set performance. The goal of this phase is a
clean, reproducible baseline -- not a maximized accuracy number.

Exposes run_training() so tests (and later Phase 2+ tooling) can invoke
the exact same code path against a temp directory / small fixture,
instead of only being runnable as a script.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.detection.if_normalization import fit_isolation_forest_normal_only
from src.observability.logging_config import get_logger, log_event
from src.preprocessing.classical_pipeline import prepare_training_data

logger = get_logger("classical_ids_training")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = REPO_ROOT / "Data" / "raw" / "KDDTrain+.txt"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "classical"
DEFAULT_PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"

# Fixed, undtuned defaults -- see module docstring.
RF_PARAMS: Dict[str, Any] = dict(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
XGB_PARAMS: Dict[str, Any] = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
    eval_metric="logloss", n_jobs=-1,
)
IF_PARAMS: Dict[str, Any] = dict(n_estimators=200, random_state=42)


@dataclass
class TrainingReport:
    dataset: Dict[str, Any]
    training_time_seconds: Dict[str, float]
    inference_time_seconds: Dict[str, float]
    metrics: Dict[str, Any]
    model_params: Dict[str, Any]


def run_training(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    preprocessing_dir: str | Path = DEFAULT_PREPROCESSING_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: Dict[str, Any] | None = None,
    xgb_params: Dict[str, Any] | None = None,
    if_params: Dict[str, Any] | None = None,
) -> TrainingReport:
    rf_params = rf_params or RF_PARAMS
    xgb_params = xgb_params or XGB_PARAMS
    if_params = if_params or IF_PARAMS

    models_dir = Path(models_dir)
    preprocessing_dir = Path(preprocessing_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    log_event(logger, 20, "Preparing training data", raw_path=str(raw_path))
    prepared = prepare_training_data(
        raw_path=raw_path,
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=preprocessing_dir,
        test_size=test_size,
        random_state=random_state,
    )
    X_train, X_test = prepared.X_train, prepared.X_test
    y_train, y_test = prepared.y_train, prepared.y_test

    # --- Random Forest ------------------------------------------------
    log_event(logger, 20, "Training Random Forest", params=rf_params)
    rf_model = RandomForestClassifier(**rf_params)
    t0 = time.perf_counter()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0

    # --- XGBoost --------------------------------------------------------
    log_event(logger, 20, "Training XGBoost", params=xgb_params)
    xgb_model = XGBClassifier(**xgb_params)
    t0 = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.perf_counter() - t0

    # --- Isolation Forest (normal-only) ---------------------------------
    log_event(logger, 20, "Training Isolation Forest", params=if_params)
    t0 = time.perf_counter()
    if_model, if_normalization = fit_isolation_forest_normal_only(
        X_train, y_train, **if_params
    )
    if_train_time = time.perf_counter() - t0

    # --- persist models + normalization ----------------------------------
    joblib.dump(rf_model, models_dir / "random_forest.joblib")
    joblib.dump(xgb_model, models_dir / "xgboost.joblib")
    joblib.dump(if_model, models_dir / "isolation_forest.joblib")
    if_normalization.save(preprocessing_dir / "isolation_forest_normalization.json")

    # --- evaluate once, on the held-out test split, using the real
    #     detector code path (not a shortcut) ---------------------------
    detector = EnsembleClassicalDetector.load(models_dir, preprocessing_dir)

    t0 = time.perf_counter()
    results = detector.detect_matrix(X_test)
    total_inference_time = time.perf_counter() - t0
    per_sample_inference_ms = (total_inference_time / len(results)) * 1000.0

    y_pred = np.array([0 if r.classical_prediction == "normal" else 1 for r in results])

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    report = TrainingReport(
        dataset={
            "raw_path": str(raw_path),
            "train_rows": prepared.train_rows,
            "test_rows": prepared.test_rows,
            "feature_count": len(prepared.feature_columns),
            "test_size": test_size,
            "random_state": random_state,
        },
        training_time_seconds={
            "random_forest": rf_train_time,
            "xgboost": xgb_train_time,
            "isolation_forest": if_train_time,
            "total": rf_train_time + xgb_train_time + if_train_time,
        },
        inference_time_seconds={
            "total": total_inference_time,
            "per_sample_ms": per_sample_inference_ms,
            "test_set_size": len(results),
        },
        metrics=metrics,
        model_params={
            "random_forest": rf_params,
            "xgboost": xgb_params,
            "isolation_forest": if_params,
        },
    )

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(asdict(report), f, indent=2)

    log_event(logger, 20, "Training complete", **metrics)
    return report


if __name__ == "__main__":
    result = run_training()
    print(json.dumps(asdict(result), indent=2))
