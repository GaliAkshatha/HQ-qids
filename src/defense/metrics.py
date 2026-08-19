"""
src/defense/metrics.py

defense_success_rate = verified healthy outcomes / actually attempted actions
    ("actually attempted" = action_status in {EXECUTED, FAILED} -- excludes
    REJECTED, since a rejected action was never attempted at all)

self_healing_success_rate = successful verified recoveries / recovery attempts
    (None if no recoveries were attempted -- avoids a misleading 0/0 -> 0.0)

Retries alone are never counted as successful self-healing -- only ones
whose independent verification confirmed the intended state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DefenseMetrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    total_defense_decisions: int = 0
    actions_selected: Dict[str, int] = field(default_factory=dict)
    actions_executed: int = 0
    actions_rejected: int = 0

    successful_remediations: int = 0
    failed_remediations: int = 0

    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    rollbacks: int = 0

    verification_failures: int = 0

    def record(
        self,
        selected_action: str,
        action_status: str,  # "EXECUTED" | "FAILED" | "REJECTED"
        health_status: str,  # "HEALTHY" | "UNHEALTHY"
        initial_verification_failed: bool,
        recovery_attempted: bool,
        recovery_status: str,  # "NOT_ATTEMPTED" | "SUCCESS" | "FAILED"
        rollback_attempted: bool,
    ) -> None:
        with self._lock:
            self.total_defense_decisions += 1
            self.actions_selected[selected_action] = self.actions_selected.get(selected_action, 0) + 1

            if action_status == "REJECTED":
                self.actions_rejected += 1
            else:
                self.actions_executed += 1
                if health_status == "HEALTHY":
                    self.successful_remediations += 1
                else:
                    self.failed_remediations += 1

            if initial_verification_failed:
                self.verification_failures += 1

            if recovery_attempted:
                self.recovery_attempts += 1
                if recovery_status == "SUCCESS":
                    self.successful_recoveries += 1
                elif recovery_status == "FAILED":
                    self.failed_recoveries += 1

            if rollback_attempted:
                self.rollbacks += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            attempted = self.actions_executed  # EXECUTED + FAILED (REJECTED excluded)
            defense_success_rate: Optional[float] = (
                self.successful_remediations / attempted if attempted else None
            )
            self_healing_success_rate: Optional[float] = (
                self.successful_recoveries / self.recovery_attempts if self.recovery_attempts else None
            )
            return {
                "total_defense_decisions": self.total_defense_decisions,
                "actions_selected": dict(self.actions_selected),
                "actions_executed": self.actions_executed,
                "actions_rejected": self.actions_rejected,
                "successful_remediations": self.successful_remediations,
                "failed_remediations": self.failed_remediations,
                "recovery_attempts": self.recovery_attempts,
                "successful_recoveries": self.successful_recoveries,
                "failed_recoveries": self.failed_recoveries,
                "rollbacks": self.rollbacks,
                "verification_failures": self.verification_failures,
                "defense_success_rate": defense_success_rate,
                "self_healing_success_rate": self_healing_success_rate,
            }
