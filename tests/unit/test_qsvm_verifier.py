import numpy as np
import pytest

from src.contracts import QuantumResult
from src.quantum.qsvm_verifier import QSVMVerifier


@pytest.fixture(scope="module")
def qsvm_verifier(trained_quantum_fixture_artifacts) -> QSVMVerifier:
    return QSVMVerifier.load(
        models_dir=trained_quantum_fixture_artifacts["qsvm_models_dir"],
        preprocessing_dir=trained_quantum_fixture_artifacts["preprocessing_dir"],
    )


@pytest.fixture(scope="module")
def sample_scaled_features(trained_quantum_fixture_artifacts):
    X_train = np.load(trained_quantum_fixture_artifacts["processed_dir"] / "X_train.npy")
    return X_train


def test_verify_returns_valid_quantum_result(qsvm_verifier, sample_scaled_features):
    result = qsvm_verifier.verify(sample_scaled_features[0], sample_id="qsvm-test-1")

    assert isinstance(result, QuantumResult)  # __post_init__ already validated
    assert result.quantum_model == "QSVM"
    assert result.status == "success"
    assert result.quantum_prediction in ("normal", "attack")
    assert 0.0 <= result.quantum_confidence <= 1.0
    assert set(result.class_probabilities.keys()) == {"normal", "attack"}
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6
    assert result.inference_time_ms is not None and result.inference_time_ms >= 0
    assert result.circuit_metadata["type"] == "ZZFeatureMap"


def test_verify_batch_matches_verify_called_per_row(qsvm_verifier, sample_scaled_features):
    batch = sample_scaled_features[:3]
    ids = ["b0", "b1", "b2"]

    batch_results = qsvm_verifier.verify_batch(batch, sample_ids=ids)
    looped_results = [qsvm_verifier.verify(batch[i], sample_id=ids[i]) for i in range(3)]

    assert len(batch_results) == 3
    for b, s in zip(batch_results, looped_results):
        assert b.sample_id == s.sample_id
        assert b.quantum_prediction == s.quantum_prediction
        assert abs(b.quantum_confidence - s.quantum_confidence) < 1e-9


def test_verify_never_raises_on_corrupted_reference_vectors(qsvm_verifier, sample_scaled_features):
    """Explicit quantum-failure-handling test: break internal state, confirm
    a failed QuantumResult is returned instead of an exception propagating."""
    original = qsvm_verifier.reference_vectors
    try:
        qsvm_verifier.reference_vectors = np.zeros((3, 999))  # shape mismatch -> will raise internally
        result = qsvm_verifier.verify(sample_scaled_features[0], sample_id="qsvm-failure-test")
        assert isinstance(result, QuantumResult)
        assert result.status == "failed"
        assert result.error is not None
        assert result.quantum_prediction is None
        assert result.quantum_confidence is None
    finally:
        qsvm_verifier.reference_vectors = original  # restore for other tests using this module-scoped fixture


def test_load_raises_clear_error_on_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        QSVMVerifier.load(models_dir=tmp_path / "nope", preprocessing_dir=tmp_path / "also_nope")
