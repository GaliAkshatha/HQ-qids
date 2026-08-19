"""
src/incident/escalation.py

Every condition here is derived from evidence already produced by
Phase 4 (RiskAssessment, HybridDecision) or Phase 5 (DefenseResult) --
nothing new is invented. repeated_incident_threshold is real, tested
code, but documented as structurally inert against NSL-KDD today (see
correlation.py) since sample_id is unique per row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.contracts import DefenseResult, HybridDecision, RiskAssessment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "incident_policy.json"


@dataclass
class EscalationPolicyConfig:
    on_critical_risk: bool
    on_unresolved_conflict_at_high_or_above: bool
    on_defense_action_failed: bool
    on_recovery_failed: bool
    repeated_incident_enabled: bool
    repeated_incident_threshold: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "EscalationPolicyConfig":
        with open(path) as f:
            raw = json.load(f)
        esc = raw["escalation"]
        return cls(
            on_critical_risk=esc["on_critical_risk"],
            on_unresolved_conflict_at_high_or_above=esc["on_unresolved_conflict_at_high_or_above"],
            on_defense_action_failed=esc["on_defense_action_failed"],
            on_recovery_failed=esc["on_recovery_failed"],
            repeated_incident_enabled=esc["repeated_incident_enabled"],
            repeated_incident_threshold=esc["repeated_incident_threshold"],
        )


def evaluate_escalation(
    risk_assessment: RiskAssessment,
    hybrid_decision: HybridDecision,
    defense_result: Optional[DefenseResult],
    repeated_incident_count: int,
    policy: EscalationPolicyConfig,
) -> Tuple[bool, List[str]]:
    """Returns (should_escalate, reason_codes). Deterministic -- same
    inputs always produce the same output, no randomness."""
    reasons: List[str] = []

    if policy.on_critical_risk and risk_assessment.risk_level == "CRITICAL":
        reasons.append("CRITICAL_RISK")

    if (
        policy.on_unresolved_conflict_at_high_or_above
        and hybrid_decision.decision_status == "uncertain"
        and risk_assessment.risk_level in ("HIGH", "CRITICAL")
    ):
        reasons.append("UNRESOLVED_QUANTUM_CONFLICT_AT_ELEVATED_RISK")

    if defense_result is not None:
        if policy.on_defense_action_failed and defense_result.action_status == "FAILED":
            reasons.append("DEFENSE_ACTION_FAILED")
        if policy.on_recovery_failed and defense_result.recovery_status == "FAILED":
            reasons.append("RECOVERY_FAILED")

    if policy.repeated_incident_enabled and repeated_incident_count >= policy.repeated_incident_threshold:
        reasons.append("REPEATED_INCIDENT_THRESHOLD_EXCEEDED")

    return len(reasons) > 0, reasons
