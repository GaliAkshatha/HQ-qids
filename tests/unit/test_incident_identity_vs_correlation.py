"""
Regression tests proving the corrected semantic distinction:

  incident_id     = individual incident identity / idempotency key
  correlation_id   = session/entity correlation key, may map to MANY
                       distinct incident_ids

This file exercises IncidentManager.record_full_lifecycle() directly
(the distributed incident-worker's own entry point) with a shared
correlation_key across multiple distinct incident_ids -- exactly the
scenario that exposed the original bug.
"""

from src.contracts import DefenseResult, DetectionResult, HybridDecision, RiskAssessment
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import InMemoryEventStore, JsonlEventStore
from src.incident.incident_manager import IncidentManager


def make_escalation_policy(repeated_threshold=3):
    return EscalationPolicyConfig(
        on_critical_risk=True, on_unresolved_conflict_at_high_or_above=True,
        on_defense_action_failed=True, on_recovery_failed=True,
        repeated_incident_enabled=True, repeated_incident_threshold=repeated_threshold,
    )


def make_manager(event_store=None, repeated_threshold=3):
    return IncidentManager(
        detector=None, router=None, hybrid_pipeline=None, defense_engine=None,
        event_store=event_store or InMemoryEventStore(),
        escalation_policy=make_escalation_policy(repeated_threshold),
    )


def make_evidence(sample_id, risk_level="LOW", decision_status="normal"):
    dr = DetectionResult(
        sample_id=sample_id, classical_prediction="normal", classical_confidence=0.9,
        class_probabilities={"normal": 0.9, "attack": 0.1}, anomaly_score=0.1, model_disagreement=0.05,
    )
    hd = HybridDecision(
        sample_id=sample_id, final_prediction="normal", final_confidence=0.9,
        quantum_used=False, decision_status=decision_status, evidence={"anomaly_score": 0.1, "model_disagreement": 0.05},
    )
    ra = RiskAssessment(
        sample_id=sample_id, risk_level=risk_level, risk_score=0.1,
        threat_evidence_score=0.1, system_uncertainty_score=0.05,
    )
    dres = DefenseResult(
        sample_id=sample_id, severity=risk_level, risk_score=0.1, action="MONITOR",
        action_status="EXECUTED", recovery_status="NOT_ATTEMPTED", health_status="HEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )
    return dr, hd, ra, dres


def _fake_routing_decision(sample_id):
    from src.contracts import RoutingDecision

    return RoutingDecision(sample_id=sample_id, decision_status="not_invoked", should_invoke_quantum=False)


def run_lifecycle(manager, correlation_key, incident_id, sample_id, **evidence_overrides):
    dr, hd, ra, dres = make_evidence(sample_id, **evidence_overrides)
    return manager.record_full_lifecycle(
        correlation_key=correlation_key, incident_id=incident_id, created_at="2026-01-01T00:00:00+00:00",
        detection_result=dr, routing_decision=_fake_routing_decision(sample_id),
        hybrid_decision=hd, risk_assessment=ra, defense_result=dres, rollback_occurred=False,
    )


# ---- Test A: same incident_id + same correlation_id submitted twice -----------

def test_a_duplicate_incident_id_is_idempotent_skip_processed_exactly_once():
    manager = make_manager()
    correlation_key = "session-A"
    incident_id = "incident-A1"

    first = run_lifecycle(manager, correlation_key, incident_id, "sample-1")
    assert first.current_state in ("RESOLVED", "ESCALATED")
    events_after_first = len(manager.get_events(incident_id))

    existing = manager.get_incident(incident_id)
    assert existing is not None and existing.is_terminal
    manager.append_idempotent_skip(existing, "redelivered incident_id")

    events_after_duplicate = manager.get_events(incident_id)
    assert len(events_after_duplicate) == events_after_first + 1
    assert events_after_duplicate[-1].event_type == "IDEMPOTENT_SKIP"

    detection_events = [e for e in events_after_duplicate if e.event_type == "DETECTION_CREATED"]
    assert len(detection_events) == 1


# ---- Test B: different incident_id + same correlation_id -> both process ------

def test_b_distinct_incident_ids_sharing_correlation_key_both_process_independently():
    manager = make_manager()
    correlation_key = "session-B"

    snap1 = run_lifecycle(manager, correlation_key, "incident-B1", "sample-1")
    snap2 = run_lifecycle(manager, correlation_key, "incident-B2", "sample-2")

    assert snap1.incident_id != snap2.incident_id
    assert snap1.current_state in ("RESOLVED", "ESCALATED")
    assert snap2.current_state in ("RESOLVED", "ESCALATED")

    assert manager.get_incident("incident-B1") is not None
    assert manager.get_incident("incident-B2") is not None

    history = manager.get_incidents_by_correlation(correlation_key)
    assert {s.incident_id for s in history} == {"incident-B1", "incident-B2"}


# ---- Test C: three distinct incidents -> repeated-incident threshold fires -----

def test_c_three_distinct_incidents_organically_trigger_repeated_incident_threshold():
    manager = make_manager(repeated_threshold=3)
    correlation_key = "session-C"

    run_lifecycle(manager, correlation_key, "incident-C1", "sample-1")
    run_lifecycle(manager, correlation_key, "incident-C2", "sample-2")
    third = run_lifecycle(manager, correlation_key, "incident-C3", "sample-3")

    assert third.escalated is True
    assert any("REPEATED_INCIDENT" in r for r in third.escalation_reasons)
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 3


# ---- Test D: duplicate delivery does NOT increment the repeated-incident counter --

def test_d_duplicate_delivery_does_not_increment_repeated_incident_counter():
    manager = make_manager(repeated_threshold=3)
    correlation_key = "session-D"

    run_lifecycle(manager, correlation_key, "incident-D1", "sample-1")
    run_lifecycle(manager, correlation_key, "incident-D2", "sample-2")
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 2

    existing = manager.get_incident("incident-D1")
    assert existing is not None and existing.is_terminal
    manager.append_idempotent_skip(existing, "redelivered")

    assert manager.repeated_incident_tracker.count_for(correlation_key) == 2

    third = run_lifecycle(manager, correlation_key, "incident-D3", "sample-3")
    assert manager.repeated_incident_tracker.count_for(correlation_key) == 3
    assert third.escalated is True


# ---- Test E: restart reconstructs history correctly ------------------------------

def test_e_restart_reconstructs_distinct_incidents_and_preserves_both_idempotency_kinds(tmp_path):
    event_store_path = tmp_path / "test_e_events.jsonl"

    store_1 = JsonlEventStore(event_store_path)
    manager_1 = make_manager(event_store=store_1, repeated_threshold=3)
    correlation_key = "session-E"

    run_lifecycle(manager_1, correlation_key, "incident-E1", "sample-1")
    run_lifecycle(manager_1, correlation_key, "incident-E2", "sample-2")
    assert manager_1.repeated_incident_tracker.count_for(correlation_key) == 2

    del manager_1
    del store_1

    store_2 = JsonlEventStore(event_store_path)
    manager_2 = make_manager(event_store=store_2, repeated_threshold=3)

    history = manager_2.get_incidents_by_correlation(correlation_key)
    assert {s.incident_id for s in history} == {"incident-E1", "incident-E2"}
    assert manager_2.repeated_incident_tracker.count_for(correlation_key) == 2

    existing = manager_2.get_incident("incident-E1")
    assert existing is not None and existing.is_terminal
    events_before = len(manager_2.get_events("incident-E1"))
    manager_2.append_idempotent_skip(existing, "redelivered after restart")
    events_after = len(manager_2.get_events("incident-E1"))
    assert events_after == events_before + 1

    third = run_lifecycle(manager_2, correlation_key, "incident-E3", "sample-3")
    assert manager_2.repeated_incident_tracker.count_for(correlation_key) == 3
    assert third.escalated is True
    assert any("REPEATED_INCIDENT" in r for r in third.escalation_reasons)
