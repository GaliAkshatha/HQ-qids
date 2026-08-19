from src.contracts import DefenseResult, DetectionResult, HybridDecision, RiskAssessment
from src.incident.escalation import EscalationPolicyConfig, evaluate_escalation


def make_policy(**overrides):
    base = dict(
        on_critical_risk=True, on_unresolved_conflict_at_high_or_above=True,
        on_defense_action_failed=True, on_recovery_failed=True,
        repeated_incident_enabled=True, repeated_incident_threshold=3,
    )
    base.update(overrides)
    return EscalationPolicyConfig(**base)


def make_risk(level="LOW"):
    return RiskAssessment(sample_id="s1", risk_level=level, risk_score=0.1, threat_evidence_score=0.1, system_uncertainty_score=0.05)


def make_hybrid(status="normal"):
    return HybridDecision(sample_id="s1", final_prediction="normal", final_confidence=0.9, quantum_used=False, decision_status=status)


def make_defense(action_status="EXECUTED", recovery_status="NOT_ATTEMPTED"):
    return DefenseResult(
        sample_id="s1", severity="LOW", risk_score=0.1, action="MONITOR",
        action_status=action_status, recovery_status=recovery_status, health_status="HEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )


def test_loads_from_real_config_file(repo_root):
    policy = EscalationPolicyConfig.load(repo_root / "config" / "incident_policy.json")
    assert policy.repeated_incident_threshold == 3
    assert policy.on_critical_risk is True


def test_no_escalation_when_nothing_triggers():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(), 1, policy)
    assert should is False
    assert reasons == []


def test_critical_risk_triggers_escalation():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("CRITICAL"), make_hybrid(), make_defense(), 1, policy)
    assert should is True
    assert "CRITICAL_RISK" in reasons


def test_critical_risk_disabled_in_config_does_not_trigger():
    policy = make_policy(on_critical_risk=False)
    should, reasons = evaluate_escalation(make_risk("CRITICAL"), make_hybrid(), make_defense(), 1, policy)
    assert should is False


def test_unresolved_conflict_at_high_risk_triggers():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("HIGH"), make_hybrid("uncertain"), make_defense(), 1, policy)
    assert should is True
    assert "UNRESOLVED_QUANTUM_CONFLICT_AT_ELEVATED_RISK" in reasons


def test_uncertain_at_low_risk_does_not_trigger_conflict_escalation():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid("uncertain"), make_defense(), 1, policy)
    assert "UNRESOLVED_QUANTUM_CONFLICT_AT_ELEVATED_RISK" not in reasons


def test_defense_action_failed_triggers():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(action_status="FAILED", recovery_status="FAILED"), 1, policy)
    assert should is True
    assert "DEFENSE_ACTION_FAILED" in reasons


def test_recovery_failed_triggers():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(action_status="FAILED", recovery_status="FAILED"), 1, policy)
    assert "RECOVERY_FAILED" in reasons


def test_recovery_success_does_not_trigger():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(action_status="EXECUTED", recovery_status="SUCCESS"), 1, policy)
    assert should is False


def test_repeated_incident_threshold_triggers_at_exact_boundary():
    policy = make_policy(repeated_incident_threshold=3)
    should_below, _ = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(), 2, policy)
    should_at, reasons_at = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(), 3, policy)
    assert should_below is False
    assert should_at is True
    assert "REPEATED_INCIDENT_THRESHOLD_EXCEEDED" in reasons_at


def test_repeated_incident_disabled_in_config():
    policy = make_policy(repeated_incident_enabled=False)
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), make_defense(), 10, policy)
    assert should is False


def test_multiple_conditions_all_recorded_simultaneously():
    policy = make_policy()
    should, reasons = evaluate_escalation(
        make_risk("CRITICAL"), make_hybrid("uncertain"),
        make_defense(action_status="FAILED", recovery_status="FAILED"), 5, policy,
    )
    assert should is True
    assert len(reasons) >= 3


def test_no_defense_result_does_not_crash():
    policy = make_policy()
    should, reasons = evaluate_escalation(make_risk("LOW"), make_hybrid(), None, 1, policy)
    assert should is False
