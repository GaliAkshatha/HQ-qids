"""
src/defense/recovery.py

Self-healing success criterion (approved, enforced literally here):
    1. initial remediation failed (verification failed)
    2. a recovery action executed (retry)
    3. simulated state reached the intended state
    4. independent verification confirmed it

self_healed is True ONLY when a retry's own independent verification
passes. Rollback succeeding is NEVER counted as self-healing -- rollback
means the intended defensive action was NOT achieved; it only means
cleanup succeeded. recovery_status="FAILED" in that case, exactly as
specified, even though rollback_succeeded may be True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.defense.executor import DefenseExecutor
from src.defense.simulated_state import SimulatedNetworkState, SourceState
from src.defense import verification as ver


@dataclass
class RecoveryResult:
    attempted: bool
    retry_count: int = 0
    retry_succeeded: bool = False
    rollback_attempted: bool = False
    rollback_succeeded: Optional[bool] = None
    self_healed: bool = False  # True ONLY if retry_succeeded (never on rollback alone)
    recovery_status: str = "NOT_ATTEMPTED"  # "NOT_ATTEMPTED" | "SUCCESS" | "FAILED"
    log: List[str] = field(default_factory=list)


def attempt_recovery(
    executor: DefenseExecutor,
    state: SimulatedNetworkState,
    action_type: str,
    target: str,
    pre_action_snapshot: SourceState,
    max_retries: int,
    rollback_on_failure: bool,
    action_reversible: bool,
) -> RecoveryResult:
    """
    Called only after the initial verification has already failed.
    Retries the same action up to max_retries times, independently
    re-verifying after each attempt. Falls back to rollback only if every
    retry still fails to verify and the action is reversible.
    """
    result = RecoveryResult(attempted=True)

    for attempt in range(1, max_retries + 1):
        result.retry_count = attempt
        exec_result = executor.execute(action_type, target)
        verification = ver.verify(state, action_type, target)
        result.log.append(f"retry {attempt}: executor.succeeded={exec_result.succeeded}, verified={verification.verified}")

        if verification.verified:
            result.retry_succeeded = True
            result.self_healed = True
            result.recovery_status = "SUCCESS"
            return result

    # every retry failed to verify
    if rollback_on_failure and action_reversible:
        result.rollback_attempted = True
        rollback_exec = executor.rollback(action_type, target, pre_action_snapshot)
        rollback_verification = ver.verify_rollback(state, target, pre_action_snapshot)
        result.rollback_succeeded = rollback_verification.verified
        result.log.append(
            f"rollback: executor.succeeded={rollback_exec.succeeded}, "
            f"independently_verified={rollback_verification.verified} ({rollback_verification.reason})"
        )

    # self_healed stays False here regardless of rollback_succeeded -- the
    # intended defensive action was never achieved, only cleanup was
    result.recovery_status = "FAILED"
    return result
