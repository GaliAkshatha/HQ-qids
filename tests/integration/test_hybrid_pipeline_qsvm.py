"""
Minimal real QSVM path through the full hybrid pipeline. Deliberately a
single sample given the real ~2.3s/sample measured cost.
"""

import dataclasses

import pytest

from src.contracts import HybridDecision, RiskAssessment
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.preprocessing.classical_pipeline import load_raw, transform_sample
from src.quantum.qsvm_verifier import QSVMVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
QSVM_MODELS = "artifacts/models/quantum/qsvm"
PREPROCESSING = "artifacts/preprocessing"


def test_single_real_qsvm_sample_through_full_hybrid_pipeline(repo_root):
    detector = EnsembleClassicalDetector.load(
        models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING
    )
    base_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    policy = base_policy.with_overrides(quantum_backend="QSVM")
    verifier = QSVMVerifier.load(models_dir=repo_root / QSVM_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    router = QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=1))

    pipeline = HybridPipeline()

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()

    detection = detector.detect(sample, sample_id="qsvm-hybrid-1")
    forced = dataclasses.replace(detection, classical_confidence=0.1)  # force routing
    scaled = transform_sample(sample, detector.preprocessing)[0]

    routing = router.route_and_wait("qsvm-hybrid-1", scaled, forced, timeout=10)
    assert routing.should_invoke_quantum is True
    assert routing.quantum_backend == "QSVM"

    hybrid_decision, risk = pipeline.process(forced, routing)

    assert isinstance(hybrid_decision, HybridDecision)
    assert isinstance(risk, RiskAssessment)
    if routing.decision_status == "success":
        assert hybrid_decision.quantum_model == "QSVM"
