"""
Minimal real QSVM path through the full pipeline into DefenseEngine.
Deliberately a single sample given the real ~2.1-2.4s/sample cost.
"""

import dataclasses

from src.contracts import DefenseResult
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
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


def test_single_real_qsvm_sample_through_full_defense_pipeline(repo_root):
    detector = EnsembleClassicalDetector.load(models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    base_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    policy = base_policy.with_overrides(quantum_backend="QSVM")
    verifier = QSVMVerifier.load(models_dir=repo_root / QSVM_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    router = QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=1))

    hybrid_pipeline = HybridPipeline()
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json"))

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()

    detection = detector.detect(sample, sample_id="qsvm-defense-1")
    forced = dataclasses.replace(detection, classical_confidence=0.1)  # force routing
    scaled = transform_sample(sample, detector.preprocessing)[0]

    routing = router.route_and_wait("qsvm-defense-1", scaled, forced, timeout=10)
    hybrid_decision, risk = hybrid_pipeline.process(forced, routing)
    defense_result = defense_engine.process(forced, hybrid_decision, risk)

    assert isinstance(defense_result, DefenseResult)
    assert defense_result.action_status in ("EXECUTED", "FAILED", "REJECTED")
    print("\nReal QSVM defense result:", defense_result.to_dict())
