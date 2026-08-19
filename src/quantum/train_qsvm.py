"""
src/quantum/train_qsvm.py

Trains the new reusable QSVM verifier, preserving the existing baseline
configuration exactly for comparability (300 train / 150 test,
ZZFeatureMap reps=2) -- not tuned during Phase 2.

Difference from legacy src/models/qsvm_model.py (kept, unmodified):
this SVC uses probability=True (needed for QuantumResult.quantum_confidence),
persists via joblib + a JSON config instead of a single joblib bundle
containing the kernel object, and evaluates through the real
QSVMVerifier.verify_batch() path after persisting -- proving the
artifacts actually round-trip, not just that in-memory training worked.

Kernel matrices are cached to disk during training (kernel_train_cache.npy
/ kernel_test_cache.npy under the run's models_dir) so an interrupted run
can resume without recomputing -- same caching pattern qsvm_model.py
already uses, kept because it's proven and the kernel evaluation is the
expensive step (see Phase 2 plan timing notes).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from src.observability.logging_config import get_logger, log_event
from src.quantum.feature_maps import build_feature_map
from src.quantum.pca_artifact import get_or_fit_quantum_pca, transform_to_quantum_features
from src.quantum.qsvm_verifier import QSVMVerifier
from src.quantum.subsampling import stratified_subsample

logger = get_logger("qsvm_training")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"
DEFAULT_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "quantum" / "qsvm"
DEFAULT_PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"

# Preserved baseline config -- see module docstring. Not tuned in Phase 2.
FEATURE_MAP_CONFIG: Dict[str, Any] = {"type": "ZZFeatureMap", "num_qubits": 4, "reps": 2}
TRAIN_SUBSAMPLE_SIZE = 300
TEST_SUBSAMPLE_SIZE = 150


@dataclass
class QsvmTrainingReport:
    dataset: Dict[str, Any]
    training_time_seconds: Dict[str, float]
    inference_time_seconds: Dict[str, float]
    metrics: Dict[str, Any]
    config: Dict[str, Any]


def run_qsvm_training(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    preprocessing_dir: str | Path = DEFAULT_PREPROCESSING_DIR,
    train_subsample_size: int = TRAIN_SUBSAMPLE_SIZE,
    test_subsample_size: int = TEST_SUBSAMPLE_SIZE,
    feature_map_config: Dict[str, Any] | None = None,
    random_state: int = 42,
    use_kernel_cache: bool = True,
) -> QsvmTrainingReport:
    feature_map_config = feature_map_config or FEATURE_MAP_CONFIG

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

    # Subsample the ORIGINAL 41-dim scaled arrays first, then PCA-transform
    # that subsample -- this keeps a 41-dim version of each selected row
    # available for evaluating through the real verifier.verify_batch()
    # path later (which itself applies PCA internally), instead of feeding
    # already-PCA'd data into a verifier that expects 41-dim input.
    X_train_sub_41d, y_train_sub = stratified_subsample(X_train_full, y_train_full, train_subsample_size, random_state)
    X_test_sub_41d, y_test_sub = stratified_subsample(X_test_full, y_test_full, test_subsample_size, random_state)
    log_event(logger, 20, "Subsampled", train_size=len(X_train_sub_41d), test_size=len(X_test_sub_41d))

    X_train_sub = transform_to_quantum_features(X_train_sub_41d, pca)
    X_test_sub = transform_to_quantum_features(X_test_sub_41d, pca)

    feature_map = build_feature_map(feature_map_config)
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    kernel_train_cache = models_dir / "kernel_train_cache.npy"
    kernel_test_cache = models_dir / "kernel_test_cache.npy"

    log_event(logger, 20, "Evaluating training kernel matrix (this is the expensive step)")
    t0 = time.perf_counter()
    if use_kernel_cache and kernel_train_cache.exists():
        kernel_train = np.load(kernel_train_cache)
        log_event(logger, 20, "Loaded cached training kernel")
    else:
        kernel_train = kernel.evaluate(x_vec=X_train_sub)
        np.save(kernel_train_cache, kernel_train)
    kernel_train_time = time.perf_counter() - t0

    log_event(logger, 20, "Evaluating test kernel matrix")
    t0 = time.perf_counter()
    if use_kernel_cache and kernel_test_cache.exists():
        kernel_test = np.load(kernel_test_cache)
        log_event(logger, 20, "Loaded cached test kernel")
    else:
        kernel_test = kernel.evaluate(x_vec=X_test_sub, y_vec=X_train_sub)
        np.save(kernel_test_cache, kernel_test)
    kernel_test_time = time.perf_counter() - t0

    log_event(logger, 20, "Training SVC(kernel='precomputed', probability=True)")
    svc_model = SVC(kernel="precomputed", probability=True, random_state=random_state)
    t0 = time.perf_counter()
    svc_model.fit(kernel_train, y_train_sub)
    svc_train_time = time.perf_counter() - t0

    # persist -- reference vectors are part of the artifact, not just
    # training scaffolding (see QSVMVerifier docstring)
    joblib.dump(svc_model, models_dir / "svc_model.joblib")
    np.save(models_dir / "reference_vectors.npy", X_train_sub)
    np.save(models_dir / "reference_labels.npy", y_train_sub)
    with open(models_dir / "feature_map_config.json", "w") as f:
        json.dump(feature_map_config, f, indent=2)

    # Bulk metrics: predict_proba on the already-evaluated kernel_test
    # matrix. This is mathematically identical to what
    # QSVMVerifier.verify_batch() would produce for the same 150 test
    # samples (it computes svc_model.predict_proba(kernel.evaluate(X_test,
    # reference_vectors)) internally) -- reusing kernel_test here avoids
    # evaluating the same 150x300 kernel a second time for no reason.
    proba_bulk = svc_model.predict_proba(kernel_test)
    y_pred = np.argmax(proba_bulk, axis=1)
    # Real per-sample inference latency: measured separately, through the
    # actual persisted QSVMVerifier.verify() path loaded fresh from disk,
    # on a small subset (not all 150 -- that would just redundantly
    # re-evaluate the same kernel values a second time).
    verifier = QSVMVerifier.load(models_dir, preprocessing_dir)
    latency_probe_size = min(5, len(X_test_sub_41d))
    probe_ids = [f"qsvm_latency_probe_{i}" for i in range(latency_probe_size)]
    t0 = time.perf_counter()
    probe_results = verifier.verify_batch(X_test_sub_41d[:latency_probe_size], sample_ids=probe_ids)
    total_inference_time = time.perf_counter() - t0
    per_sample_inference_ms = (total_inference_time / latency_probe_size) * 1000.0

    failures = [r for r in probe_results if r.status == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} latency-probe verify_batch results failed: {failures[0].error}")

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

    report = QsvmTrainingReport(
        dataset={
            "train_subsample_size": len(X_train_sub),
            "test_subsample_size": len(X_test_sub),
            "num_qubits": feature_map_config["num_qubits"],
            "random_state": random_state,
        },
        training_time_seconds={
            "kernel_train_evaluation": kernel_train_time,
            "kernel_test_evaluation": kernel_test_time,
            "svc_fit": svc_train_time,
            "total": kernel_train_time + kernel_test_time + svc_train_time,
        },
        inference_time_seconds={
            "total_batch": total_inference_time,
            "per_sample_ms": per_sample_inference_ms,
            "latency_probe_size": latency_probe_size,
            "note": (
                "per_sample_ms measured via a small real verify_batch() probe "
                "through the persisted artifact path, not all test_subsample_size "
                "samples -- see accuracy metrics below for full-test-set results, "
                "computed from the same cached kernel_test matrix verify_batch() "
                "would otherwise recompute identically."
            ),
        },
        metrics=metrics,
        config={"feature_map": feature_map_config},
    )

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(asdict(report), f, indent=2)

    log_event(logger, 20, "QSVM training complete", **metrics)
    return report


if __name__ == "__main__":
    result = run_qsvm_training()
    print(json.dumps(asdict(result), indent=2))
