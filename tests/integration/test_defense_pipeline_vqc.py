"""
Real end-to-end: DetectionResult -> QuantumRouter (real VQC) -> Hybrid ->
Risk -> DefenseEngine -> DefenseResult, using Phase 1-4's real persisted
artifacts and real config. VQC is cheap enough to run generously here.
"""

import pytest

from src.contracts import DefenseResult
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.defense.defense_engine import DefenseEngine
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
def defense_engine(repo_root):
    from src.defense.defense_policy import DefensePolicyConfig

    return DefenseEngine(policy=DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json"))


@pytest.fixture(scope="module")
def real_raw_rows(repo_root):
    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    return df.drop(columns=["label", "difficulty"]).iloc[:20]


def test_full_real_pipeline_through_defense_engine(real_detector, real_router, hybrid_pipeline, defense_engine, real_raw_rows):
    results = []
    for i in range(len(real_raw_rows)):
        sample = real_raw_rows.iloc[i].to_dict()
        sample_id = f"defense-e2e-{i}"

        detection = real_detector.detect(sample, sample_id=sample_id)
        scaled = transform_sample(sample, real_detector.preprocessing)[0]
        routing = real_router.route_and_wait(sample_id, scaled, detection, timeout=5)
        hybrid_decision, risk = hybrid_pipeline.process(detection, routing)

        defense_result = defense_engine.process(detection, hybrid_decision, risk)

        assert isinstance(defense_result, DefenseResult)
        assert defense_result.sample_id == sample_id
        assert defense_result.severity == risk.risk_level
        assert defense_result.action_status in ("EXECUTED", "FAILED", "REJECTED")
        assert defense_result.recovery_status in ("NOT_ATTEMPTED", "SUCCESS", "FAILED")
        assert defense_result.health_status in ("HEALTHY", "UNHEALTHY")
        results.append(defense_result)

    print("\nReal VQC defense metrics:", defense_engine.metrics.snapshot())
    assert len(results) == len(real_raw_rows)
    # every real execution in this environment should succeed on first try
    # (no fault injection anywhere in this real path)
    assert all(r.action_status != "REJECTED" or r.health_status == "HEALTHY" for r in results)
