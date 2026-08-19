"""
src/quantum/train_vqc.py

Trains the new reusable VQC verifier, preserving the existing baseline
configuration exactly for comparability (200 train / 100 test,
RealAmplitudes reps=2, COBYLA maxiter=100) -- not tuned during Phase 2.

Difference from legacy src/models/vqc_model.py (kept, unmodified): this
persists via to_dill()/from_dill() instead of raw pickle, and evaluates
through the real VQCVerifier.verify_batch() path after persisting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from qiskit_machine_learning.algorithms import VQC
from sklearn.metrics import confusion_matrix

from src.observability.logging_config import get_logger, log_event
from src.quantum.feature_maps import build_ansatz, build_feature_map, build_optimizer
from src.quantum.pca_artifact import get_or_fit_quantum_pca, transform_to_quantum_features
from src.quantum.subsampling import stratified_subsample
from src.quantum.vqc_verifier import VQCVerifier

logger = get_logger("vqc_training")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "quantum" / "vqc"
DEFAULT_PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"

# Preserved baseline config -- see module docstring. Not tuned in Phase 2.
FEATURE_MAP_CONFIG: Dict[str, Any] = {"type": "ZZFeatureMap", "num_qubits": 4, "reps": 2}
ANSATZ_CONFIG: Dict[str, Any] = {"type": "RealAmplitudes", "num_qubits": 4, "reps": 2}
OPTIMIZER_CONFIG: Dict[str, Any] = {"type": "COBYLA", "maxiter": 100}
TRAIN_SUBSAMPLE_SIZE = 200
TEST_SUBSAMPLE_SIZE = 100


@dataclass
class VqcTrainingReport:
    dataset: Dict[str, Any]
    training_time_seconds: Dict[str, float]
    inference_time_seconds: Dict[str, float]
    metrics: Dict[str, Any]
    config: Dict[str, Any]


def run_vqc_training(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    preprocessing_dir: str | Path = DEFAULT_PREPROCESSING_DIR,
    train_subsample_size: int = TRAIN_SUBSAMPLE_SIZE,
    test_subsample_size: int = TEST_SUBSAMPLE_SIZE,
    feature_map_config: Dict[str, Any] | None = None,
    ansatz_config: Dict[str, Any] | None = None,
    optimizer_config: Dict[str, Any] | None = None,
    random_state: int = 42,
) -> VqcTrainingReport:
    feature_map_config = feature_map_config or FEATURE_MAP_CONFIG
    ansatz_config = ansatz_config or ANSATZ_CONFIG
    optimizer_config = optimizer_config or OPTIMIZER_CONFIG

    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    preprocessing_dir = Path(preprocessing_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    X_train_full = np.load(processed_dir / "X_train.npy")
    X_test_full = np.load(processed_dir / "X_test.npy")
    y_train_full = np.load(processed_dir / "y_train.npy")
    y_test_full = np.load(processed_dir / "y_test.npy")

    log_event(logger, 20, "Fitting/loading shared quantum PCA")
    pca = get_or_fit_quantum_pca(X_train_full, preprocessing_dir, n_components=feature_map_config["num_qubits"])

    X_train_sub_41d, y_train_sub = stratified_subsample(X_train_full, y_train_full, train_subsample_size, random_state)
    X_test_sub_41d, y_test_sub = stratified_subsample(X_test_full, y_test_full, test_subsample_size, random_state)
    log_event(logger, 20, "Subsampled", train_size=len(X_train_sub_41d), test_size=len(X_test_sub_41d))

    X_train_sub = transform_to_quantum_features(X_train_sub_41d, pca)

    feature_map = build_feature_map(feature_map_config)
    ansatz = build_ansatz(ansatz_config)
    optimizer = build_optimizer(optimizer_config)

    vqc_model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)

    log_event(logger, 20, "Training VQC", optimizer=optimizer_config)
    t0 = time.perf_counter()
    vqc_model.fit(X_train_sub, y_train_sub)
    train_time = time.perf_counter() - t0

    vqc_model.to_dill(str(models_dir / "vqc_model.dill"))
    with open(models_dir / "feature_map_config.json", "w") as f:
        json.dump(feature_map_config, f, indent=2)
    with open(models_dir / "ansatz_config.json", "w") as f:
        json.dump(ansatz_config, f, indent=2)
    with open(models_dir / "optimizer_config.json", "w") as f:
        json.dump(optimizer_config, f, indent=2)

    # evaluate through the REAL verifier path, loaded fresh, not a
    # shortcut -- feed the 41-dim subsample, matching what the verifier
    # actually expects (it applies PCA internally)
    verifier = VQCVerifier.load(models_dir, preprocessing_dir)
    test_ids = [f"vqc_eval_{i}" for i in range(len(X_test_sub_41d))]

    t0 = time.perf_counter()
    results = verifier.verify_batch(X_test_sub_41d, sample_ids=test_ids)
    total_inference_time = time.perf_counter() - t0
    per_sample_inference_ms = (total_inference_time / len(results)) * 1000.0

    failures = [r for r in results if r.status == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} verify_batch results failed during evaluation: {failures[0].error}")

    y_pred = np.array([0 if r.quantum_prediction == "normal" else 1 for r in results])

    tn, fp, fn, tp = confusion_matrix(y_test_sub, y_pred, labels=[0, 1]).ravel()
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

    report = VqcTrainingReport(
        dataset={
            "train_subsample_size": len(X_train_sub),
            "test_subsample_size": len(X_test_sub_41d),
            "num_qubits": feature_map_config["num_qubits"],
            "random_state": random_state,
        },
        training_time_seconds={"vqc_fit": train_time, "total": train_time},
        inference_time_seconds={
            "total_batch": total_inference_time,
            "per_sample_ms": per_sample_inference_ms,
            "test_set_size": len(results),
        },
        metrics=metrics,
        config={
            "feature_map": feature_map_config,
            "ansatz": ansatz_config,
            "optimizer": optimizer_config,
        },
    )

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(asdict(report), f, indent=2)

    log_event(logger, 20, "VQC training complete", **metrics)
    return report


if __name__ == "__main__":
    result = run_vqc_training()
    print(json.dumps(asdict(result), indent=2))
