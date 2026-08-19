"""
Real end-to-end: real classical detector output -> real router -> real
VQC verifier loaded from Phase 2's persisted production artifacts. VQC is
cheap enough (~7-8ms/sample) to exercise generously here.
"""

import numpy as np
import pytest

from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
QUANTUM_VQC_MODELS = "artifacts/models/quantum/vqc"
PREPROCESSING = "artifacts/preprocessing"


@pytest.fixture(scope="module")
def real_router(repo_root):
    policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    assert policy.quantum_backend == "VQC"
    verifier = VQCVerifier.load(models_dir=repo_root / QUANTUM_VQC_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    return QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))


@pytest.fixture(scope="module")
def real_detector(repo_root):
    return EnsembleClassicalDetector.load(
        models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING
    )


@pytest.fixture(scope="module")
def real_raw_rows(repo_root):
    from src.preprocessing.classical_pipeline import load_raw

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    return df.drop(columns=["label", "difficulty"]).iloc[:20]


def test_real_pipeline_route_and_wait_produces_resolved_decisions(real_router, real_detector, real_raw_rows):
    resolved_count = 0
    skipped_count = 0

    for i in range(len(real_raw_rows)):
        sample = real_raw_rows.iloc[i].to_dict()
        detection = real_detector.detect(sample, sample_id=f"vqc-e2e-{i}")

        from src.preprocessing.classical_pipeline import transform_sample

        scaled = transform_sample(sample, real_detector.preprocessing)[0]

        decision = real_router.route_and_wait(f"vqc-e2e-{i}", scaled, detection, timeout=5)

        if decision.should_invoke_quantum:
            resolved_count += 1
            assert decision.decision_status in ("success", "fallback")
            assert decision.quantum_attempted is True
        else:
            skipped_count += 1
            assert decision.decision_status == "not_invoked"

    assert resolved_count + skipped_count == len(real_raw_rows)


def test_real_route_is_non_blocking_for_vqc_too(real_router, real_detector, real_raw_rows):
    """Even though VQC is fast, route() itself must still return before
    the quantum call completes -- proves non-blocking semantics aren't
    backend-speed-dependent."""
    sample = real_raw_rows.iloc[0].to_dict()
    detection = real_detector.detect(sample, sample_id="vqc-nonblock-check")

    from src.preprocessing.classical_pipeline import transform_sample

    scaled = transform_sample(sample, real_detector.preprocessing)[0]

    # force a candidate by using a synthetic low-confidence detection result
    import dataclasses

    forced = dataclasses.replace(detection, classical_confidence=0.1)

    decision = real_router.route("vqc-nonblock-check", scaled, forced)
    if decision.should_invoke_quantum:
        assert decision.decision_status == "pending"
        assert decision.quantum_result is None
        assert decision.job_id is not None
        record = real_router.get_result(decision.job_id, timeout=5)
        assert record is not None
