"""
src/defense/verification.py

Verification deliberately does NOT accept the executor's ExecutionResult
as proof of anything. It takes only (action_type, target) and reads
SimulatedNetworkState fresh, itself -- so "the executor said succeeded"
can never substitute for "we independently confirmed the state actually
changed." This is the literal enforcement of that requirement, not just
a naming convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from src.defense import action_catalog as ac
from src.defense.simulated_state import SimulatedNetworkState, SourceState


@dataclass
class VerificationResult:
    verified: bool
    action_type: str
    target: str
    checked_fields: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""


def _expected_state_checks(action_type: str, source: SourceState) -> Dict[str, bool]:
    """Returns {field_description: whether it matches the action's intended effect}."""
    if action_type == ac.MONITOR:
        return {"monitored": source.monitored is True}
    if action_type == ac.INCREASE_MONITORING:
        return {"monitoring_level_increased": source.monitoring_level == "INCREASED"}
    if action_type == ac.RATE_LIMIT:
        return {"rate_limited": source.rate_limited is True}
    if action_type == ac.ISOLATE_SIMULATED_SOURCE:
        return {
            "isolated": source.isolated is True,
            "sessions_suspended": source.active_sessions == 0 and source.suspended_sessions > 0,
        }
    if action_type == ac.BLOCK_SIMULATED_SOURCE:
        return {
            "blocked": source.blocked is True,
            "rate_limited_as_side_effect": source.rate_limited is True,
            "sessions_dropped": source.active_sessions == 0,
        }
    if action_type == ac.TERMINATE_SIMULATED_SESSION:
        return {"terminated": source.terminated is True, "sessions_cleared": source.active_sessions == 0}
    raise ValueError(f"No verification rule defined for action type: '{action_type}'")


def verify(state: SimulatedNetworkState, action_type: str, target: str) -> VerificationResult:
    """Independently re-reads current state for `target` and checks it
    against the action's intended effect. Called after both initial
    execution AND after any recovery/rollback attempt -- always the same
    independent check, never a different "trust it this time" path."""
    source = state.get(target)
    if source is None:
        return VerificationResult(verified=False, action_type=action_type, target=target, reason=f"no state exists for target '{target}'")

    checks = _expected_state_checks(action_type, source)
    verified = all(checks.values())
    reason = "all expected state fields confirmed" if verified else f"expected state not reached: {checks}"
    return VerificationResult(verified=verified, action_type=action_type, target=target, checked_fields=checks, reason=reason)


def verify_rollback(state: SimulatedNetworkState, target: str, expected_snapshot: SourceState) -> VerificationResult:
    """Independently confirms rollback restored the COMPLETE pre-action
    state, not just one field -- compares every field of the current
    state against the snapshot taken before the original action."""
    source = state.get(target)
    if source is None:
        return VerificationResult(verified=False, action_type="ROLLBACK", target=target, reason=f"no state exists for target '{target}'")

    matches = source == expected_snapshot
    checks = {
        "blocked": source.blocked == expected_snapshot.blocked,
        "isolated": source.isolated == expected_snapshot.isolated,
        "rate_limited": source.rate_limited == expected_snapshot.rate_limited,
        "monitoring_level": source.monitoring_level == expected_snapshot.monitoring_level,
        "active_sessions": source.active_sessions == expected_snapshot.active_sessions,
        "suspended_sessions": source.suspended_sessions == expected_snapshot.suspended_sessions,
        "terminated": source.terminated == expected_snapshot.terminated,
    }
    reason = "rollback fully matches pre-action snapshot" if matches else f"rollback incomplete: {checks}"
    return VerificationResult(verified=matches, action_type="ROLLBACK", target=target, checked_fields=checks, reason=reason)
