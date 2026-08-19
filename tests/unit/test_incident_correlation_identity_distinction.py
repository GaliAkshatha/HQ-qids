"""
tests/unit/test_incident_correlation_identity_distinction.py

Proves the Phase 8-driven fix to IncidentManager: incident_id is the
idempotency identity (a redelivered incident_id must be skipped, never
reprocessed); correlation_id is the grouping/session identity (many
distinct incident_ids may legitimately share one correlation_id, each
processed independently, with the shared key used for repeated-incident
counting and session history).

Uses IncidentManager.record_full_lifecycle() directly with real contract
objects (mirroring how the distributed incident-worker calls it) so
these tests exercise the exact code path Phase 8 actually broke, not a
synthetic stand-in.
"""

from src.contracts import DefenseResult, DetectionResult, HybridDecision, RiskAssessment
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import InMemoryEventStore, JsonlEventStore
from src.incident.incident_manager import IncidentManager


def make_manager(event_store=None, escalation_policy=None):
    return IncidentManager(
        detector=None, router=None, hybrid_pipeline=None, defense_engine=None,
        event_store=event_store or InMemoryEventStore(),
        escalation_policy=escalation_policy or EscalationPolicyConfig(
            on_critical_risk=True, on_unresolved_conflict_at_high_or_above=True,
            on_defense_action_failed=True, on_recovery_failed=True,
            repeated_incident_enabled=True, repeated_incident_threshold=3,
        ),
    )


def make_evidence(sample_id, risk_level="LOW", decision_status="normal"):
    dr = DetectionResult(
        sample_id=sample_id, classical_prediction="attack" if risk_level != "LOW" else "normal",
        classical_confidence=0.9, class_probabilities={"normal": 0.1, "attack": 0.9},
        anomaly_score=0.5, model_disagreement=0.1,
    )
    hd = HybridDecision(
        sample_id=sample_id, final_prediction=dr.classical_prediction, final_confidence=0.9,
        quantum_used=False, decision_status=decision_status, evidence={"anomaly_score": 0.5, "model_disagreement": 0.1},
    )
    ra = RiskAssessment(
        sample_id=sample_id, risk_level=risk_level, risk_score=0.3,
        threat_evidence_score=0.3, system_uncertainty_score=0.1,
    )
    dres = DefenseResult(
        sample_id=sample_id, severity=risk_level, risk_score=0.3, action="MONITOR",
        action_status="EXECUTED", recovery_status="NOT_ATTEMPTED", health_status="HEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )
    return dr, hd, ra, dres


class _FakeRoutingDecision:
    def __init__(self, sample_id):
        self.sample_id = sample_id
        self.should_invoke_quantum = False
        self.reason_codes = []
        self.decision_status = "not_invoked"
        self.fallback_reason = None
        self.quantum_backend = None
        self.quantum_result = None


def process_incident(manager, correlation_key, incident_id, sample_id, risk_level="LOW", decision_status="normal"):
    dr, hd, ra, dres = make_evidence(sample_id, risk_level, decision_status)
    return manager.record_full_lifecycle(
        correlation_key=correlation_key, incident_id=incident_id, created_at="2026-01-01T00:00:00+00:00",
        detection_result=dr, routing_decision=_FakeRoutingDecision(sample_id), hybrid_decision=hd,
        risk_assessment=ra, defense_result=dres, rollback_occurred=False,
    )


# ---- Test A: same incident_id + same correlation_id, delivered twice ------------

def test_a_duplicate_incident_id_is_idempotent_skip_processed_exactly_once():
    manager = make_manager()
    correlation_key = "session-A"
    incident_id = "inc-A-1"

    first = process_incident(manager, correlation_key, incident_id, "sample-1")
    assert first.current_state == "RESOLVED"

    existing = manager.get_incident(incident_id)
    assert existing is not None and existing.is_terminal
    manager.append_idempotent_skip(existing, "duplicate delivery test")

    events = manager.get_events(incident_id)
    event_types = [e.event_type for e in events]
    assert event_types.count("DETECTION_CREATED") == 1
    assert event_types.count("IDEMPOTENT_SKIP") == 1
    assert manager.metrics.snapshot()["total_incidents"] == 1


# ---- Test B: different incident_id + same correlation_id -> both process ---------

def test_b_different_incident_ids_same_correlation_id_both_process_independently():
    manager = make_manager()
    correlation_key = "session-B"

    result1 = process_incident(manager, correlation_key, "inc-B-1", "sample-1")
    result2 = process_incident(manager, correlation_key, "inc-B-2", "sample-2")

    assert result1.incident_id == "inc-B-1"
    assert result2.incident_id == "inc-B-2"
    assert result1.current_state == "RESOLVED"
    assert result2.current_state == "RESOLVED"

    assert manager.get_incident("inc-B-1") is not None
    assert manager.get_incident("inc-B-2") is not None

    history = manager.get_incidents_by_correlation(correlation_key)
    assert {s.incident_id for s in history} == {"inc-B-1", "inc-B-2"}

    assert manager.metrics.snapshot()["total_incidents"] == 2


# ---- Test C: three distinct incident_ids, same correlation_id -> threshold fires ---

def test_c_three_distinct_incidents_same_correlation_id_triggers_repeated_threshold():
    manager = make_manager()  # threshold=3
    correlation_key = "session-C"

    r1 = process_incident(manager, correlation_key, "inc-C-1", "sample-1")
    r2 = process_incident(manager, correlation_key, "inc-C-2", "sample-2")
    r3 = process_incident(manager, correlation_key, "inc-C-3", "sample-3")

    assert r1.escalated is False
    assert r2.escalated is False
    assert r3.escalated is True
    assert "REPEATED_INCIDENT_THRESHOLD_EXCEEDED" in r3.escalation_reasons

    assert manager.repeated_incident_tracker.count_for(correlation_key) == 3


# ---- Test D: duplicate delivery of one of those three does NOT double-count -------

def test_d_duplicate_delivery_does_not_increment_repeated_incident_counter():
    manager = make_manager()
    correlation_key = "session-D"

    process_incident(manager, correlation_key, "inc-D-1", "sample-1")
    process_incident(manager, correlation_key, "inc-D-2", "sample-2")
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 2

    existing = manager.get_incident("inc-D-1")
    manager.append_idempotent_skip(existing, "duplicate redelivery")
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 2  # unchanged

    r3 = process_incident(manager, correlation_key, "inc-D-3", "sample-3")
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 3
    assert r3.escalated is True


# ---- Test E: restart -- reconstruct, duplicate stays idempotent, new incident counted ---

def test_e_restart_preserves_identity_distinction(tmp_path):
    store_path = tmp_path / "events.jsonl"
    correlation_key = "session-E"

    store_1 = JsonlEventStore(store_path)
    manager_1 = make_manager(event_store=store_1)
    process_incident(manager_1, correlation_key, "inc-E-1", "sample-1")
    process_incident(manager_1, correlation_key, "inc-E-2", "sample-2")
    assert manager_1.repeated_incident_tracker.count_for(correlation_key) == 2
    del manager_1
    del store_1

    store_2 = JsonlEventStore(store_path)
    manager_2 = make_manager(event_store=store_2)

    assert manager_2.get_incident("inc-E-1") is not None
    assert manager_2.get_incident("inc-E-2") is not None
    assert manager_2.repeated_incident_tracker.count_for(correlation_key) == 2

    existing = manager_2.get_incident("inc-E-1")
    assert existing.is_terminal
    manager_2.append_idempotent_skip(existing, "post-restart duplicate")
    assert manager_2.repeated_incident_tracker.count_for(correlation_key) == 2  # unchanged

    r3 = process_incident(manager_2, correlation_key, "inc-E-3", "sample-3")
    assert r3.escalated is True
    assert manager_2.repeated_incident_tracker.count_for(correlation_key) == 3


# ---- Backward compatibility: SampleIdCorrelation keeps 1:1 semantics -------------

def test_backward_compat_sample_id_correlation_remains_one_to_one():
    """Under the default SampleIdCorrelation (unchanged from Phase 1-7),
    correlation_key IS sample_id, so this fix must not change any
    existing single-incident-per-key behavior."""
    manager = make_manager()
    result = process_incident(manager, "sample-xyz", "inc-xyz", "sample-xyz")
    assert manager.get_incident_by_correlation("sample-xyz").incident_id == result.incident_id
    assert len(manager.get_incidents_by_correlation("sample-xyz")) == 1
