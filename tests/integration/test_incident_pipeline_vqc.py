"""
Real end-to-end: DetectionResult -> QuantumRouter (real VQC) -> Hybrid ->
Risk -> Defense -> IncidentManager, using Phase 1-5's real persisted
artifacts and real config. VQC is cheap enough to run generously here.
"""

import pytest

from src.contracts import IncidentSnapshot
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.decision_policy import DecisionPolicyConfig
from src.hybrid.pipeline import HybridPipeline
from src.hybrid.risk_policy import RiskPolicyConfig
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import InMemoryEventStore
from src.incident.incident_manager import IncidentManager
from src.preprocessing.classical_pipeline import load_raw
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
VQC_MODELS = "artifacts/models/quantum/vqc"
PREPROCESSING = "artifacts/preprocessing"


@pytest.fixture(scope="module")
def real_manager(repo_root):
    detector = EnsembleClassicalDetector.load(models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    routing_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    verifier = VQCVerifier.load(models_dir=repo_root / VQC_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    router = QuantumRouter(policy=routing_policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))
    hybrid_pipeline = HybridPipeline(
        decision_policy=DecisionPolicyConfig.load(repo_root / "config" / "hybrid_decision_policy.json"),
        risk_policy=RiskPolicyConfig.load(repo_root / "config" / "risk_policy.json"),
    )
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json"))
    escalation_policy = EscalationPolicyConfig.load(repo_root / "config" / "incident_policy.json")

    return IncidentManager(
        detector=detector, router=router, hybrid_pipeline=hybrid_pipeline, defense_engine=defense_engine,
        event_store=InMemoryEventStore(), escalation_policy=escalation_policy,
    )


@pytest.fixture(scope="module")
def real_raw_rows(repo_root):
    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    return df.drop(columns=["label", "difficulty"]).iloc[:20]


def test_full_real_phase1_to_6_pipeline(real_manager, real_raw_rows):
    snapshots = []
    for i in range(len(real_raw_rows)):
        sample = real_raw_rows.iloc[i].to_dict()
        sample_id = f"incident-e2e-{i}"
        snapshot = real_manager.process(sample_id, sample)

        assert isinstance(snapshot, IncidentSnapshot)
        assert snapshot.is_terminal
        assert snapshot.current_state in ("RESOLVED", "ESCALATED")

        events = real_manager.get_events(snapshot.incident_id)
        assert len(events) > 0
        assert events[0].event_type == "DETECTION_CREATED"
        assert events[-1].event_type in ("INCIDENT_RESOLVED", "INCIDENT_ESCALATED")

        # event ordering is monotonic (timestamps non-decreasing)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

        snapshots.append(snapshot)

    print("\nReal VQC incident metrics:", real_manager.metrics_snapshot())
    assert len(snapshots) == len(real_raw_rows)


def test_incident_lookup_apis_work_on_real_data(real_manager, real_raw_rows):
    sample = real_raw_rows.iloc[0].to_dict()
    snapshot = real_manager.process("incident-lookup-check", sample)

    by_id = real_manager.get_incident(snapshot.incident_id)
    by_correlation = real_manager.get_incident_by_correlation("incident-lookup-check")
    events = real_manager.get_events(snapshot.incident_id)

    assert by_id is not None and by_id.incident_id == snapshot.incident_id
    assert by_correlation is not None and by_correlation.incident_id == snapshot.incident_id
    assert len(events) == len(snapshot.event_ids)
