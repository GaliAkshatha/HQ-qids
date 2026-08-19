"""
src/incident/metrics.py

Incident-level metrics, distinct granularity from RouterMetrics/
HybridMetrics/DefenseMetrics (those measure per-call outcomes; this
measures per-incident lifecycle outcomes). active_incidents will
typically read 0 between calls in Phase 6's synchronous design (an
incident runs start-to-terminal within one IncidentManager.process()
call) -- documented honestly, not hidden.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class IncidentMetrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    total_incidents: int = 0
    active_incidents: int = 0
    resolved_incidents: int = 0
    escalated_incidents: int = 0

    quantum_verification_count: int = 0
    quantum_failure_count: int = 0

    defense_failures: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    rollback_count: int = 0

    escalation_count: int = 0

    _resolution_times_seconds: List[float] = field(default_factory=list, repr=False)

    def record_incident_created(self) -> None:
        with self._lock:
            self.total_incidents += 1
            self.active_incidents += 1

    def record_quantum_verification(self, failed: bool) -> None:
        with self._lock:
            self.quantum_verification_count += 1
            if failed:
                self.quantum_failure_count += 1

    def record_defense_outcome(self, failed: bool) -> None:
        with self._lock:
            if failed:
                self.defense_failures += 1

    def record_recovery_attempt(self, succeeded: bool, rollback_occurred: bool) -> None:
        with self._lock:
            self.recovery_attempts += 1
            if succeeded:
                self.recovery_successes += 1
            if rollback_occurred:
                self.rollback_count += 1

    def record_incident_terminal(self, escalated: bool, created_at_iso: str, resolved_at_iso: str) -> None:
        with self._lock:
            self.active_incidents -= 1
            if escalated:
                self.escalated_incidents += 1
                self.escalation_count += 1
            else:
                self.resolved_incidents += 1
            try:
                created = datetime.fromisoformat(created_at_iso)
                resolved = datetime.fromisoformat(resolved_at_iso)
                self._resolution_times_seconds.append((resolved - created).total_seconds())
            except ValueError:
                pass  # malformed timestamp -- skip rather than corrupt the average

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            mean_resolution: Optional[float] = (
                sum(self._resolution_times_seconds) / len(self._resolution_times_seconds)
                if self._resolution_times_seconds else None
            )
            return {
                "total_incidents": self.total_incidents,
                "active_incidents": self.active_incidents,
                "resolved_incidents": self.resolved_incidents,
                "escalated_incidents": self.escalated_incidents,
                "quantum_verification_count": self.quantum_verification_count,
                "quantum_failure_count": self.quantum_failure_count,
                "defense_failures": self.defense_failures,
                "recovery_attempts": self.recovery_attempts,
                "recovery_successes": self.recovery_successes,
                "rollback_count": self.rollback_count,
                "escalation_count": self.escalation_count,
                "mean_resolution_time_seconds": mean_resolution,
            }
