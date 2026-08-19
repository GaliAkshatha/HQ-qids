import dataclasses

from src.contracts import DefenseResult, DetectionResult, HybridDecision, RiskAssessment
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.defense.simulated_state import SimulatedNetworkState


def make_policy(**overrides):
    base = dict(
        risk_action_map={
            "LOW": {"normal": "MONITOR", "confirmed": "MONITOR", "uncertain": "MONITOR"},
            "MEDIUM": {"normal": "INCREASE_MONITORING", "confirmed": "INCREASE_MONITORING", "uncertain": "INCREASE_MONITORING"},
            "HIGH": {"normal": "RATE_LIMIT", "confirmed": "ISOLATE_SIMULATED_SOURCE", "uncertain": "RATE_LIMIT"},
            "CRITICAL": {"normal": "ISOLATE_SIMULATED_SOURCE", "confirmed": "BLOCK_SIMULATED_SOURCE", "uncertain": "ISOLATE_SIMULATED_SOURCE"},
        },
        disabled_actions=[], uncertain_requires_reversible_action=True,
        max_recovery_retries=1, rollback_on_failure=True, simulation_mode=True,
    )
    base.update(overrides)
    return DefensePolicyConfig(**base)


def make_evidence(sample_id="s1", risk_level="HIGH", decision_status="confirmed"):
    dr = DetectionResult(
        sample_id=sample_id, classical_prediction="attack", classical_confidence=0.9,
        class_probabilities={"normal": 0.1, "attack": 0.9}, anomaly_score=0.8, model_disagreement=0.1,
    )
    hd = HybridDecision(
        sample_id=sample_id, final_prediction="attack", final_confidence=0.9,
        quantum_used=False, decision_status=decision_status, evidence={"anomaly_score": 0.8, "model_disagreement": 0.1},
    )
    ra = RiskAssessment(
        sample_id=sample_id, risk_level=risk_level, risk_score=0.6,
        threat_evidence_score=0.7, system_uncertainty_score=0.1,
    )
    return dr, hd, ra


def test_successful_defense_execution():
    engine = DefenseEngine(policy=make_policy())
    dr, hd, ra = make_evidence()

    result = engine.process(dr, hd, ra)

    assert isinstance(result, DefenseResult)
    assert result.action == "ISOLATE_SIMULATED_SOURCE"
    assert result.action_status == "EXECUTED"
    assert result.health_status == "HEALTHY"
    assert result.recovery_status == "NOT_ATTEMPTED"
    assert result.severity == "HIGH"
    assert result.rollback_available is True

    metrics = engine.metrics.snapshot()
    assert metrics["actions_executed"] == 1
    assert metrics["successful_remediations"] == 1


def test_failed_defense_execution_with_exhausted_recovery():
    state = SimulatedNetworkState()
    state.inject_failure_for("s1")  # permanent fault -- retry will also fail
    engine = DefenseEngine(policy=make_policy(), state=state)
    dr, hd, ra = make_evidence()

    result = engine.process(dr, hd, ra)

    assert result.action_status == "FAILED"
    assert result.health_status == "UNHEALTHY"
    assert result.recovery_status == "FAILED"

    metrics = engine.metrics.snapshot()
    assert metrics["failed_remediations"] == 1
    assert metrics["recovery_attempts"] == 1
    assert metrics["failed_recoveries"] == 1
    assert metrics["rollbacks"] == 1


def test_rejected_action_never_reaches_executor():
    engine = DefenseEngine(policy=make_policy(simulation_mode=False))
    dr, hd, ra = make_evidence()

    result = engine.process(dr, hd, ra)

    assert result.action_status == "REJECTED"
    assert result.recovery_status == "NOT_ATTEMPTED"
    metrics = engine.metrics.snapshot()
    assert metrics["actions_rejected"] == 1
    assert metrics["actions_executed"] == 0


def test_idempotent_repeated_processing_of_same_sample_is_rejected_second_time():
    engine = DefenseEngine(policy=make_policy())
    dr, hd, ra = make_evidence()

    first = engine.process(dr, hd, ra)
    assert first.action_status == "EXECUTED"

    second = engine.process(dr, hd, ra)  # same sample, same action already active
    assert second.action_status == "REJECTED"
    assert second.health_status == "HEALTHY"  # already in the intended protective state


def test_low_risk_normal_uses_monitor_and_succeeds():
    engine = DefenseEngine(policy=make_policy())
    dr, hd, ra = make_evidence(risk_level="LOW", decision_status="normal")

    result = engine.process(dr, hd, ra)
    assert result.action == "MONITOR"
    assert result.action_status == "EXECUTED"


def test_every_generated_defense_result_passes_existing_contract_validation():
    engine = DefenseEngine(policy=make_policy())
    cases = [
        make_evidence("s1", "LOW", "normal"),
        make_evidence("s2", "MEDIUM", "uncertain"),
        make_evidence("s3", "HIGH", "confirmed"),
        make_evidence("s4", "CRITICAL", "confirmed"),
    ]
    for dr, hd, ra in cases:
        result = engine.process(dr, hd, ra)
        # re-construct from to_dict() to prove it round-trips through the
        # contract's own unmodified validation
        DefenseResult(**result.to_dict())


def test_metrics_track_actions_by_type():
    engine = DefenseEngine(policy=make_policy())
    engine.process(*make_evidence("s1", "LOW", "normal"))
    engine.process(*make_evidence("s2", "LOW", "normal"))
    engine.process(*make_evidence("s3", "MEDIUM", "normal"))

    snap = engine.metrics.snapshot()
    assert snap["actions_selected"]["MONITOR"] == 2
    assert snap["actions_selected"]["INCREASE_MONITORING"] == 1
    assert snap["defense_success_rate"] == 1.0
