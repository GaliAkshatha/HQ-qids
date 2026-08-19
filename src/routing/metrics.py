"""
src/routing/metrics.py

In-process, thread-safe counters for the router. Persistence to a
database is explicitly Phase 7 scope (persistent incident/event history)
-- this is the in-memory + logged observability layer for Phase 3.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RouterMetrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    total_events: int = 0
    quantum_candidates: int = 0  # should_invoke_quantum == True
    skipped_events: int = 0  # should_invoke_quantum == False
    quantum_invocations: int = 0  # actually submitted (circuit allowed it)
    quantum_successes: int = 0
    quantum_failures: int = 0
    fallback_count: int = 0  # circuit_open + timeout + retries_exhausted, combined
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    _routing_latencies_ms: List[float] = field(default_factory=list, repr=False)
    _quantum_latencies_ms: Dict[str, List[float]] = field(default_factory=dict, repr=False)

    def record_event(self, should_invoke: bool, routing_latency_ms: float) -> None:
        with self._lock:
            self.total_events += 1
            self._routing_latencies_ms.append(routing_latency_ms)
            if should_invoke:
                self.quantum_candidates += 1
            else:
                self.skipped_events += 1

    def record_quantum_invocation(self) -> None:
        with self._lock:
            self.quantum_invocations += 1

    def record_quantum_outcome(self, backend: str, success: bool, execution_time_ms: Optional[float]) -> None:
        with self._lock:
            if success:
                self.quantum_successes += 1
            else:
                self.quantum_failures += 1
            if execution_time_ms is not None:
                self._quantum_latencies_ms.setdefault(backend, []).append(execution_time_ms)

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self.fallback_count += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            def _stats(values: List[float]) -> Dict[str, Optional[float]]:
                if not values:
                    return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
                sorted_vals = sorted(values)
                return {
                    "count": len(values),
                    "mean_ms": statistics.mean(values),
                    "p50_ms": statistics.median(values),
                    "p95_ms": sorted_vals[int(0.95 * (len(sorted_vals) - 1))],
                    "max_ms": max(values),
                }

            quantum_latency_by_backend = {
                backend: _stats(values) for backend, values in self._quantum_latencies_ms.items()
            }

            return {
                "total_events": self.total_events,
                "quantum_candidates": self.quantum_candidates,
                "skipped_events": self.skipped_events,
                "quantum_invocations": self.quantum_invocations,
                "quantum_successes": self.quantum_successes,
                "quantum_failures": self.quantum_failures,
                "fallback_count": self.fallback_count,
                "fallback_reasons": dict(self.fallback_reasons),
                "quantum_invocation_rate": (
                    self.quantum_invocations / self.total_events if self.total_events else 0.0
                ),
                "quantum_success_rate": (
                    self.quantum_successes / self.quantum_invocations if self.quantum_invocations else 0.0
                ),
                "quantum_failure_rate": (
                    self.quantum_failures / self.quantum_invocations if self.quantum_invocations else 0.0
                ),
                "fallback_rate": (
                    self.fallback_count / self.quantum_candidates if self.quantum_candidates else 0.0
                ),
                "routing_latency_ms": _stats(self._routing_latencies_ms),
                "quantum_latency_ms_by_backend": quantum_latency_by_backend,
            }
