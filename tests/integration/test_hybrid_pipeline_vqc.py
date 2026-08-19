"""
Real end-to-end: DetectionResult -> QuantumRouter (real VQC) -> resolved
RoutingDecision -> HybridDecision -> RiskAssessment, using Phase 1-3's
real persisted artifacts. VQC is cheap enough to run generously here.
"""

import numpy as np
import pytest

from src.contracts import HybridDecision, RiskAssessment
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.preprocessing.classical_pipeline import load_raw, transform_sample
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
VQC_MODELS = "artifacts/models/quantum/vqc"
PREPROCESSING = "artifacts/preprocessing"


@pytest.fixture(scope="module")
def real_detector(repo_root):
    return EnsembleClassicalDetector.load(models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING)


@pytest.fixture(scope="module")
def real_router(repo_root):
    policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    assert policy.quantum_backend == "VQC"
    verifier = VQCVerifier.load(models_dir=repo_root / VQC_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    return QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))


@pytest.fixture(scope="module")
def hybrid_pipeline(repo_root):
    from src.hybrid.decision_policy import DecisionPolicyConfig
    from src.hybrid.risk_policy import RiskPolicyConfig

    return HybridPipeline(
        decision_policy=DecisionPolicyConfig.load(repo_root / "config" / "hybrid_decision_policy.json"),
        risk_policy=RiskPolicyConfig.load(repo_root / "config" / "risk_policy.json"),
    )


@pytest.fixture(scope="module")
def real_raw_rows(repo_root):
    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    return df.drop(columns=["label", "difficulty"]).iloc[:25]


def test_full_real_pipeline_produces_valid_hybrid_decisions_and_risk(
    real_detector, real_router, hybrid_pipeline, real_raw_rows
):
    for i in range(len(real_raw_rows)):
        sample = real_raw_rows.iloc[i].to_dict()
        sample_id = f"hybrid-e2e-{i}"

        detection = real_detector.detect(sample, sample_id=sample_id)
        scaled = transform_sample(sample, real_detector.preprocessing)[0]
        routing = real_router.route_and_wait(sample_id, scaled, detection, timeout=5)

        hybrid_decision, risk = hybrid_pipeline.process(detection, routing)

        assert isinstance(hybrid_decision, HybridDecision)
        assert isinstance(risk, RiskAssessment)
        assert hybrid_decision.final_prediction in ("normal", "attack")
        assert risk.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert hybrid_decision.sample_id == risk.sample_id == sample_id

    snap = hybrid_pipeline.metrics_snapshot()
    assert snap["total_samples"] == len(real_raw_rows)
    print("\nReal VQC hybrid pipeline metrics:", snap)


def test_forced_low_confidence_sample_actually_invokes_quantum_and_resolves(
    real_detector, real_router, hybrid_pipeline, real_raw_rows
):
    import dataclasses

    sample = real_raw_rows.iloc[0].to_dict()
    detection = real_detector.detect(sample, sample_id="hybrid-forced-1")
    forced = dataclasses.replace(detection, classical_confidence=0.1)
    scaled = transform_sample(sample, real_detector.preprocessing)[0]

    routing = real_router.route_and_wait("hybrid-forced-1", scaled, forced, timeout=5)
    assert routing.should_invoke_quantum is True
    assert routing.decision_status in ("success", "fallback")

    hybrid_decision, risk = hybrid_pipeline.process(forced, routing)
    if routing.decision_status == "success":
        assert hybrid_decision.quantum_used is True
        assert hybrid_decision.verification_reason  # non-empty, includes routing + outcome codes
