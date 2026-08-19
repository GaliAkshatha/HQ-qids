"""
Real end-to-end with the real QSVM verifier. Deliberately limited to 2
real samples -- QSVM inference is genuinely ~2.1-2.4s/sample (Phase 2
measurement), so this proves the real integration works without making
the test suite prohibitively slow.
"""

import dataclasses

import pytest

from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.preprocessing.classical_pipeline import load_raw, transform_sample
from src.quantum.qsvm_verifier import QSVMVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
QUANTUM_QSVM_MODELS = "artifacts/models/quantum/qsvm"
PREPROCESSING = "artifacts/preprocessing"


@pytest.fixture(scope="module")
def qsvm_router(repo_root):
    base_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    policy = base_policy.with_overrides(quantum_backend="QSVM")
    verifier = QSVMVerifier.load(models_dir=repo_root / QUANTUM_QSVM_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    return QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=2))


@pytest.fixture(scope="module")
def real_detector(repo_root):
    return EnsembleClassicalDetector.load(
        models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING
    )


def test_real_qsvm_route_and_wait_two_samples(qsvm_router, real_detector, repo_root):
    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    raw_features = df.drop(columns=["label", "difficulty"]).iloc[:2]

    for i in range(len(raw_features)):
        sample = raw_features.iloc[i].to_dict()
        detection = real_detector.detect(sample, sample_id=f"qsvm-e2e-{i}")
        # force routing so we genuinely exercise the real QSVM call
        forced = dataclasses.replace(detection, classical_confidence=0.1)
        scaled = transform_sample(sample, real_detector.preprocessing)[0]

        decision = qsvm_router.route_and_wait(f"qsvm-e2e-{i}", scaled, forced, timeout=10)

        assert decision.should_invoke_quantum is True
        assert decision.quantum_backend == "QSVM"
        assert decision.decision_status in ("success", "fallback")
        assert decision.quantum_attempted is True
        assert decision.quantum_execution_time_ms is not None
        assert decision.total_quantum_job_time_ms is not None


def test_real_qsvm_route_is_non_blocking(qsvm_router, real_detector, repo_root):
    """The critical system-design property: route() must return almost
    immediately even though the real QSVM call underneath takes seconds."""
    import time

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[2].to_dict()
    detection = real_detector.detect(sample, sample_id="qsvm-nonblock")
    forced = dataclasses.replace(detection, classical_confidence=0.1)
    scaled = transform_sample(sample, real_detector.preprocessing)[0]

    t0 = time.perf_counter()
    decision = qsvm_router.route("qsvm-nonblock", scaled, forced)
    elapsed = time.perf_counter() - t0

    assert decision.should_invoke_quantum is True
    assert decision.decision_status == "pending"
    assert decision.quantum_result is None
    assert elapsed < 0.5  # nowhere near the real ~2.1-2.4s QSVM call

    # clean up: wait for the real job so it doesn't leak past the test
    qsvm_router.get_result(decision.job_id, timeout=10)
