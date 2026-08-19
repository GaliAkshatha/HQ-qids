from src.incident.metrics import IncidentMetrics


def test_incident_lifecycle_counts():
    metrics = IncidentMetrics()
    metrics.record_incident_created()
    metrics.record_incident_created()
    metrics.record_incident_terminal(escalated=False, created_at_iso="2026-01-01T00:00:00+00:00", resolved_at_iso="2026-01-01T00:00:05+00:00")
    metrics.record_incident_terminal(escalated=True, created_at_iso="2026-01-01T00:00:00+00:00", resolved_at_iso="2026-01-01T00:00:10+00:00")

    snap = metrics.snapshot()
    assert snap["total_incidents"] == 2
    assert snap["active_incidents"] == 0
    assert snap["resolved_incidents"] == 1
    assert snap["escalated_incidents"] == 1
    assert snap["escalation_count"] == 1
    assert snap["mean_resolution_time_seconds"] == 7.5  # (5 + 10) / 2


def test_mean_resolution_time_is_none_with_no_terminal_incidents():
    metrics = IncidentMetrics()
    metrics.record_incident_created()
    snap = metrics.snapshot()
    assert snap["mean_resolution_time_seconds"] is None
    assert snap["active_incidents"] == 1


def test_quantum_and_recovery_counters():
    metrics = IncidentMetrics()
    metrics.record_quantum_verification(failed=False)
    metrics.record_quantum_verification(failed=True)
    metrics.record_recovery_attempt(succeeded=True, rollback_occurred=False)
    metrics.record_recovery_attempt(succeeded=False, rollback_occurred=True)
    metrics.record_defense_outcome(failed=True)

    snap = metrics.snapshot()
    assert snap["quantum_verification_count"] == 2
    assert snap["quantum_failure_count"] == 1
    assert snap["recovery_attempts"] == 2
    assert snap["recovery_successes"] == 1
    assert snap["rollback_count"] == 1
    assert snap["defense_failures"] == 1
