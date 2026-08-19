"""
src/defense/safety_validator.py

Every rule here must pass before an action is allowed to reach the
executor. simulation_mode is checked first and is a hard kill-switch --
if it is ever False, validation fails unconditionally and nothing
downstream runs, regardless of any other input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.contracts import DetectionResult, HybridDecision, RiskAssessment
from src.defense import action_catalog as ac
from src.defense.defense_policy import DefensePolicyConfig
from src.defense.simulated_state import SimulatedNetworkState


@dataclass
class ValidationResult:
    allowed: bool
    reason: str
    checks: Dict[str, bool] = field(default_factory=dict)


def _already_active(action_type: str, state: SimulatedNetworkState, target: str) -> bool:
    source = state.get(target)
    if source is None:
        return False
    if action_type == ac.RATE_LIMIT:
        return source.rate_limited
    if action_type == ac.ISOLATE_SIMULATED_SOURCE:
        return source.isolated
    if action_type == ac.BLOCK_SIMULATED_SOURCE:
        return source.blocked
    if action_type == ac.INCREASE_MONITORING:
        return source.monitoring_level == "INCREASED"
    if action_type == ac.TERMINATE_SIMULATED_SESSION:
        return source.terminated
    return False  # MONITOR is always safe to (re)apply


class DefenseSafetyValidator:
    def __init__(self, policy: DefensePolicyConfig, state: SimulatedNetworkState) -> None:
        self.policy = policy
        self.state = state

    def validate(
        self,
        action_type: str,
        target: str,
        risk_level: str,
        decision_status: str,
        detection_result: DetectionResult,
        hybrid_decision: HybridDecision,
        risk_assessment: RiskAssessment,
    ) -> ValidationResult:
        checks: Dict[str, bool] = {}

        # 1. hard kill-switch, checked first, unconditionally
        checks["simulation_mode_enabled"] = self.policy.simulation_mode
        if not self.policy.simulation_mode:
            return ValidationResult(allowed=False, reason="simulation_mode is disabled -- no defense action may execute", checks=checks)

        # 2. evidence consistency -- all inputs must reference the same sample
        sample_ids = {detection_result.sample_id, hybrid_decision.sample_id, risk_assessment.sample_id}
        checks["evidence_consistent"] = len(sample_ids) == 1
        if not checks["evidence_consistent"]:
            return ValidationResult(allowed=False, reason=f"sample_id mismatch across evidence: {sample_ids}", checks=checks)

        # 3. action matches what policy actually maps for this risk/status pair
        expected_action = self.policy.select_action(risk_level, decision_status)
        checks["action_matches_policy"] = action_type == expected_action
        if not checks["action_matches_policy"]:
            return ValidationResult(
                allowed=False,
                reason=f"action '{action_type}' does not match policy-mapped action '{expected_action}' for risk_level={risk_level}, decision_status={decision_status}",
                checks=checks,
            )

        # 4. action type known and supported
        checks["action_known"] = ac.is_known_action(action_type)
        if not checks["action_known"]:
            return ValidationResult(allowed=False, reason=f"unknown action type: '{action_type}'", checks=checks)

        # 5. action not disabled by policy
        checks["action_not_disabled"] = action_type not in self.policy.disabled_actions
        if not checks["action_not_disabled"]:
            return ValidationResult(allowed=False, reason=f"action '{action_type}' is disabled by policy", checks=checks)

        # 6. target is valid
        checks["target_valid"] = bool(target)
        if not checks["target_valid"]:
            return ValidationResult(allowed=False, reason="target is empty/invalid", checks=checks)

        # 7. not already active (idempotency)
        already_active = _already_active(action_type, self.state, target)
        checks["not_already_active"] = not already_active
        if already_active:
            return ValidationResult(allowed=False, reason=f"action '{action_type}' is already active for target '{target}'", checks=checks)

        # 8. reversibility required for uncertain decisions
        spec = ac.get_action_spec(action_type)
        if decision_status == "uncertain" and self.policy.uncertain_requires_reversible_action:
            checks["reversible_if_uncertain"] = spec.reversible
            if not spec.reversible:
                return ValidationResult(
                    allowed=False,
                    reason=f"decision_status='uncertain' requires a reversible action, but '{action_type}' is not reversible",
                    checks=checks,
                )
        else:
            checks["reversible_if_uncertain"] = True  # not applicable

        return ValidationResult(allowed=True, reason="all safety checks passed", checks=checks)
