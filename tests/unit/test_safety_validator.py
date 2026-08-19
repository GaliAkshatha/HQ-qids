import dataclasses

from src.contracts import DetectionResult, HybridDecision, RiskAssessment
from src.defense.defense_policy import DefensePolicyConfig
from src.defense.safety_validator import DefenseSafetyValidator
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


def make_evidence(sample_id="s1"):
    dr = DetectionResult(
        sample_id=sample_id, classical_prediction="attack", classical_confidence=0.9,
        class_probabilities={"normal": 0.1, "attack": 0.9}, anomaly_score=0.8, model_disagreement=0.1,
    )
    hd = HybridDecision(
        sample_id=sample_id, final_prediction="attack", final_confidence=0.9,
        quantum_used=False, decision_status="confirmed", evidence={"anomaly_score": 0.8, "model_disagreement": 0.1},
    )
    ra = RiskAssessment(
        sample_id=sample_id, risk_level="HIGH", risk_score=0.6,
        threat_evidence_score=0.7, system_uncertainty_score=0.1,
    )
    return dr, hd, ra


def test_valid_action_is_allowed():
    policy = make_policy()
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()

    result = validator.validate("ISOLATE_SIMULATED_SOURCE", "s1", "HIGH", "confirmed", dr, hd, ra)
    assert result.allowed is True
    assert result.checks["simulation_mode_enabled"] is True


def test_simulation_mode_disabled_blocks_everything():
    policy = make_policy(simulation_mode=False)
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()

    result = validator.validate("MONITOR", "s1", "LOW", "normal", dr, hd, ra)
    assert result.allowed is False
    assert "simulation_mode" in result.reason
    assert result.checks["simulation_mode_enabled"] is False


def test_mismatched_evidence_sample_ids_rejected():
    policy = make_policy()
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence(sample_id="s1")
    hd = dataclasses.replace(hd, sample_id="different-sample")

    result = validator.validate("ISOLATE_SIMULATED_SOURCE", "s1", "HIGH", "confirmed", dr, hd, ra)
    assert result.allowed is False
    assert "mismatch" in result.reason


def test_action_not_matching_policy_rejected():
    policy = make_policy()
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()

    # policy maps HIGH+confirmed -> ISOLATE_SIMULATED_SOURCE, not BLOCK
    result = validator.validate("BLOCK_SIMULATED_SOURCE", "s1", "HIGH", "confirmed", dr, hd, ra)
    assert result.allowed is False
    assert "does not match policy" in result.reason


def test_unknown_action_rejected():
    policy = make_policy()
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()
    ra_low = dataclasses.replace(ra, risk_level="LOW")

    result = validator.validate("NOT_A_REAL_ACTION", "s1", "LOW", "confirmed", dr, hd, ra_low)
    assert result.allowed is False


def test_disabled_action_rejected():
    policy = make_policy(disabled_actions=["MONITOR"])
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()
    ra_low = dataclasses.replace(ra, risk_level="LOW")
    hd_normal = dataclasses.replace(hd, decision_status="normal")

    result = validator.validate("MONITOR", "s1", "LOW", "normal", dr, hd_normal, ra_low)
    assert result.allowed is False
    assert "disabled" in result.reason


def test_empty_target_rejected():
    policy = make_policy()
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()

    result = validator.validate("ISOLATE_SIMULATED_SOURCE", "", "HIGH", "confirmed", dr, hd, ra)
    assert result.allowed is False
    assert "target" in result.reason


def test_already_active_action_rejected_idempotency():
    policy = make_policy()
    state = SimulatedNetworkState()
    source = state.get_or_create("s1")
    source.isolated = True  # already isolated
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()

    result = validator.validate("ISOLATE_SIMULATED_SOURCE", "s1", "HIGH", "confirmed", dr, hd, ra)
    assert result.allowed is False
    assert result.checks["not_already_active"] is False


def test_uncertain_requires_reversible_action():
    policy = make_policy(
        risk_action_map={
            "LOW": {"normal": "MONITOR", "confirmed": "MONITOR", "uncertain": "MONITOR"},
            "MEDIUM": {"normal": "INCREASE_MONITORING", "confirmed": "INCREASE_MONITORING", "uncertain": "INCREASE_MONITORING"},
            "HIGH": {"normal": "RATE_LIMIT", "confirmed": "ISOLATE_SIMULATED_SOURCE", "uncertain": "TERMINATE_SIMULATED_SESSION"},  # deliberately irreversible for uncertain
            "CRITICAL": {"normal": "ISOLATE_SIMULATED_SOURCE", "confirmed": "BLOCK_SIMULATED_SOURCE", "uncertain": "ISOLATE_SIMULATED_SOURCE"},
        },
    )
    state = SimulatedNetworkState()
    validator = DefenseSafetyValidator(policy, state)
    dr, hd, ra = make_evidence()
    hd_uncertain = dataclasses.replace(hd, decision_status="uncertain")
    ra_high = dataclasses.replace(ra, risk_level="HIGH")

    result = validator.validate("TERMINATE_SIMULATED_SESSION", "s1", "HIGH", "uncertain", dr, hd_uncertain, ra_high)
    assert result.allowed is False
    assert result.checks["reversible_if_uncertain"] is False
