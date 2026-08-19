"""
Regression test: simulation_mode=False must result in ZERO defense
execution, unconditionally, across every risk level and decision_status
combination -- not just the common case.
"""

from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.defense.simulated_state import SimulatedNetworkState
from tests.unit.test_defense_engine import make_evidence


def make_disabled_policy():
    return DefensePolicyConfig(
        risk_action_map={
            "LOW": {"normal": "MONITOR", "confirmed": "MONITOR", "uncertain": "MONITOR"},
            "MEDIUM": {"normal": "INCREASE_MONITORING", "confirmed": "INCREASE_MONITORING", "uncertain": "INCREASE_MONITORING"},
            "HIGH": {"normal": "RATE_LIMIT", "confirmed": "ISOLATE_SIMULATED_SOURCE", "uncertain": "RATE_LIMIT"},
            "CRITICAL": {"normal": "ISOLATE_SIMULATED_SOURCE", "confirmed": "BLOCK_SIMULATED_SOURCE", "uncertain": "ISOLATE_SIMULATED_SOURCE"},
        },
        disabled_actions=[], uncertain_requires_reversible_action=True,
        max_recovery_retries=1, rollback_on_failure=True,
        simulation_mode=False,  # THE kill-switch, disabled
    )


def test_simulation_mode_false_rejects_every_risk_level_and_status_combination():
    state = SimulatedNetworkState()
    engine = DefenseEngine(policy=make_disabled_policy(), state=state)

    combinations = [
        ("s1", "LOW", "normal"), ("s2", "LOW", "confirmed"), ("s3", "LOW", "uncertain"),
        ("s4", "MEDIUM", "normal"), ("s5", "MEDIUM", "confirmed"), ("s6", "MEDIUM", "uncertain"),
        ("s7", "HIGH", "normal"), ("s8", "HIGH", "confirmed"), ("s9", "HIGH", "uncertain"),
        ("s10", "CRITICAL", "normal"), ("s11", "CRITICAL", "confirmed"), ("s12", "CRITICAL", "uncertain"),
    ]

    for sample_id, risk_level, decision_status in combinations:
        dr, hd, ra = make_evidence(sample_id=sample_id, risk_level=risk_level, decision_status=decision_status)
        result = engine.process(dr, hd, ra)
        assert result.action_status == "REJECTED", f"expected REJECTED for {risk_level}/{decision_status}, got {result.action_status}"
        assert result.recovery_status == "NOT_ATTEMPTED"

    # ZERO state mutation occurred across all 12 combinations -- the
    # simulated state was never touched by the executor at all
    for sample_id, _, _ in combinations:
        source = state.get(sample_id)
        assert source is None, f"target '{sample_id}' should never have been created in simulated state"

    snap = engine.metrics.snapshot()
    assert snap["actions_rejected"] == 12
    assert snap["actions_executed"] == 0
    assert snap["successful_remediations"] == 0
    assert snap["failed_remediations"] == 0
