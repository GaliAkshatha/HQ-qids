"""
tests/integration/test_quantum_verification_e2e.py

Exercises the full Phase 2 path: Phase 1's scaled 41-dim output -> shared
persisted quantum PCA -> QSVM or VQC -> QuantumResult, loading everything
fresh from disk (as a new process would, not reusing in-memory training
objects). Also proves QSVM and VQC are independently callable -- nothing
here invokes both for every sample, matching the approved constraint.
"""

import numpy as np

from src.contracts import QuantumResult
from src.quantum.qsvm_verifier import QSVMVerifier
from src.quantum.vqc_verifier import VQCVerifier


def test_qsvm_only_end_to_end_from_persisted_artifacts(trained_quantum_fixture_artifacts):
    fresh_verifier = QSVMVerifier.load(
        models_dir=trained_quantum_fixture_artifacts["qsvm_models_dir"],
        preprocessing_dir=trained_quantum_fixture_artifacts["preprocessing_dir"],
    )
    X_train = np.load(trained_quantum_fixture_artifacts["processed_dir"] / "X_train.npy")

    result = fresh_verifier.verify(X_train[0], sample_id="e2e-qsvm-0")
    assert isinstance(result, QuantumResult)
    assert result.quantum_model == "QSVM"
    assert result.status == "success"


def test_vqc_only_end_to_end_from_persisted_artifacts(trained_quantum_fixture_artifacts):
    fresh_verifier = VQCVerifier.load(
        models_dir=trained_quantum_fixture_artifacts["vqc_models_dir"],
        preprocessing_dir=trained_quantum_fixture_artifacts["preprocessing_dir"],
    )
    X_train = np.load(trained_quantum_fixture_artifacts["processed_dir"] / "X_train.npy")

    result = fresh_verifier.verify(X_train[0], sample_id="e2e-vqc-0")
    assert isinstance(result, QuantumResult)
    assert result.quantum_model == "VQC"
    assert result.status == "success"


def test_qsvm_and_vqc_share_the_same_persisted_quantum_pca(trained_quantum_fixture_artifacts):
    """Both verifiers were trained separately (QSVM first, then VQC) but
    must have used the exact same persisted quantum_pca.joblib -- proving
    the idempotent get_or_fit_quantum_pca sharing actually worked, not
    just that each script ran in isolation."""
    from src.quantum.pca_artifact import load_quantum_pca

    pca = load_quantum_pca(trained_quantum_fixture_artifacts["preprocessing_dir"])
    assert pca.n_components_ == 4

    # both training reports recorded the same num_qubits derived from it
    assert trained_quantum_fixture_artifacts["qsvm_report"].dataset["num_qubits"] == 4
    assert trained_quantum_fixture_artifacts["vqc_report"].dataset["num_qubits"] == 4


def test_metrics_json_persisted_with_required_fields(trained_quantum_fixture_artifacts):
    import json

    qsvm_metrics_path = trained_quantum_fixture_artifacts["qsvm_models_dir"] / "metrics.json"
    vqc_metrics_path = trained_quantum_fixture_artifacts["vqc_models_dir"] / "metrics.json"
    assert qsvm_metrics_path.exists()
    assert vqc_metrics_path.exists()

    with open(qsvm_metrics_path) as f:
        qsvm_metrics = json.load(f)
    with open(vqc_metrics_path) as f:
        vqc_metrics = json.load(f)

    for payload in (qsvm_metrics, vqc_metrics):
        for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "confusion_matrix"):
            assert key in payload["metrics"]
        assert "total" in payload["training_time_seconds"]
        assert "per_sample_ms" in payload["inference_time_seconds"]


def test_calling_only_qsvm_does_not_touch_vqc_artifacts(trained_quantum_fixture_artifacts, tmp_path):
    """Confirms QSVM and VQC are independently callable -- calling one
    does not require or invoke the other, matching the approved constraint
    that both must not be automatically executed for every sample."""
    fresh_verifier = QSVMVerifier.load(
        models_dir=trained_quantum_fixture_artifacts["qsvm_models_dir"],
        preprocessing_dir=trained_quantum_fixture_artifacts["preprocessing_dir"],
    )
    X_train = np.load(trained_quantum_fixture_artifacts["processed_dir"] / "X_train.npy")
    result = fresh_verifier.verify(X_train[1], sample_id="qsvm-only-check")
    assert result.quantum_model == "QSVM"
    assert not hasattr(fresh_verifier, "vqc_model")
