"""
Minimal real QSVM path through the full IncidentManager chain.
Deliberately a single sample given the real ~2.1-2.4s/sample cost.
"""

import dataclasses

from src.contracts import IncidentSnapshot
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import InMemoryEventStore
from src.incident.incident_manager import IncidentManager
from src.preprocessing.classical_pipeline import load_raw
from src.quantum.qsvm_verifier import QSVMVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
QSVM_MODELS = "artifacts/models/quantum/qsvm"
PREPROCESSING = "artifacts/preprocessing"


def test_single_real_qsvm_sample_through_incident_manager(repo_root):
    detector = EnsembleClassicalDetector.load(models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    base_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    policy = base_policy.with_overrides(quantum_backend="QSVM")
    verifier = QSVMVerifier.load(models_dir=repo_root / QSVM_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    router = QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=1))

    hybrid_pipeline = HybridPipeline()
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json"))
    escalation_policy = EscalationPolicyConfig.load(repo_root / "config" / "incident_policy.json")

    manager = IncidentManager(
        detector=detector, router=router, hybrid_pipeline=hybrid_pipeline, defense_engine=defense_engine,
        event_store=InMemoryEventStore(), escalation_policy=escalation_policy,
    )

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()

    snapshot = manager.process("qsvm-incident-1", sample)

    assert isinstance(snapshot, IncidentSnapshot)
    assert snapshot.is_terminal
    events = manager.get_events(snapshot.incident_id)
    print("\nReal QSVM incident event sequence:", [e.event_type for e in events])
    print("Final state:", snapshot.current_state)
