import pytest

from src.contracts import IncidentEvent, IncidentSnapshot
from src.contracts.incident import ASSESSING, DETECTED, DETECTION_CREATED, ESCALATED, INCIDENT_ESCALATED, INCIDENT_RESOLVED, MITIGATING, RESOLVED
from src.incident.incident_state import reconstruct_snapshot


def make_event(event_id="e1", incident_id="inc-1", correlation_id="s1", event_type=DETECTION_CREATED,
                previous_state=DETECTED, new_state=DETECTED, reason="ok", payload=None):
    return IncidentEvent(
        event_id=event_id, correlation_id=correlation_id, incident_id=incident_id, event_type=event_type,
        previous_state=previous_state, new_state=new_state, timestamp="2026-01-01T00:00:00+00:00",
        reason=reason, payload=payload or {},
    )


def test_event_serialization_round_trip():
    event = make_event(payload={"classical_prediction": "attack", "nested": {"a": 1}})
    data = event.to_dict()
    reloaded = IncidentEvent.from_dict(data)
    assert reloaded == event


def test_event_rejects_unknown_event_type():
    with pytest.raises(ValueError):
        make_event(event_type="NOT_A_REAL_EVENT_TYPE")


def test_event_rejects_unknown_state():
    with pytest.raises(ValueError):
        make_event(previous_state="NOT_A_STATE")


def test_snapshot_serialization_round_trip():
    snap = IncidentSnapshot(
        incident_id="inc-1", correlation_id="s1", current_state=MITIGATING,
        created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:01:00+00:00",
        event_ids=["e1", "e2"], escalated=False, escalation_reasons=[],
    )
    data = snap.to_dict()
    reloaded = IncidentSnapshot.from_dict(data)
    assert reloaded == snap


def test_snapshot_rejects_unknown_state():
    with pytest.raises(ValueError):
        IncidentSnapshot(
            incident_id="inc-1", correlation_id="s1", current_state="NOT_A_STATE",
            created_at="t", updated_at="t",
        )


def test_snapshot_requires_reasons_when_escalated():
    with pytest.raises(ValueError):
        IncidentSnapshot(
            incident_id="inc-1", correlation_id="s1", current_state=ESCALATED,
            created_at="t", updated_at="t", escalated=True, escalation_reasons=[],
        )


def test_is_terminal_property():
    resolved = IncidentSnapshot(incident_id="i", correlation_id="s", current_state=RESOLVED, created_at="t", updated_at="t")
    active = IncidentSnapshot(incident_id="i", correlation_id="s", current_state=MITIGATING, created_at="t", updated_at="t")
    assert resolved.is_terminal is True
    assert active.is_terminal is False


def test_reconstruct_snapshot_from_ordered_events_matches_direct_construction():
    events = [
        make_event(event_id="e1", event_type=DETECTION_CREATED, previous_state=DETECTED, new_state=DETECTED),
        make_event(event_id="e2", event_type=DETECTION_CREATED, previous_state=DETECTED, new_state=ASSESSING),
        make_event(event_id="e3", event_type=DETECTION_CREATED, previous_state=ASSESSING, new_state=MITIGATING),
        make_event(event_id="e4", event_type=INCIDENT_RESOLVED, previous_state=MITIGATING, new_state=RESOLVED, reason="done"),
    ]
    snapshot = reconstruct_snapshot("inc-1", events)
    assert snapshot.current_state == RESOLVED
    assert snapshot.event_ids == ["e1", "e2", "e3", "e4"]
    assert snapshot.is_terminal is True
    assert snapshot.escalated is False


def test_reconstruct_snapshot_captures_escalation_reasons():
    events = [
        make_event(event_id="e1", event_type=DETECTION_CREATED, previous_state=DETECTED, new_state=DETECTED),
        make_event(event_id="e2", event_type=INCIDENT_ESCALATED, previous_state=DETECTED, new_state=ESCALATED, reason="CRITICAL_RISK"),
    ]
    snapshot = reconstruct_snapshot("inc-1", events)
    assert snapshot.current_state == ESCALATED
    assert snapshot.escalated is True
    assert "CRITICAL_RISK" in snapshot.escalation_reasons


def test_reconstruct_snapshot_returns_none_for_unknown_incident():
    assert reconstruct_snapshot("does-not-exist", []) is None
