"""
src/routing/circuit_breaker.py

Standard three-state circuit breaker, keyed per backend name (a small
dict internally) even though only one backend is active per config in
Phase 3 -- cheap forward-compatibility for later dynamic backend
selection, not overengineering.

Deterministic and testable: time comes from an injectable now_fn
(defaults to time.monotonic), so tests can fake cooldown elapsing without
a real sleep. Thread-safe: record_success()/record_failure() are called
from worker threads, allow_request() from the caller's thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"


@dataclass
class _BreakerState:
    state: str = CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_trial_in_flight: bool = False


class QuantumCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._now_fn = now_fn
        self._states: Dict[str, _BreakerState] = {}
        self._lock = threading.Lock()

    def _state_for(self, backend: str) -> _BreakerState:
        if backend not in self._states:
            self._states[backend] = _BreakerState()
        return self._states[backend]

    def state(self, backend: str) -> str:
        with self._lock:
            return self._state_for(backend).state

    def allow_request(self, backend: str) -> bool:
        """
        True if a request should be allowed through right now. For OPEN,
        this is also where the OPEN -> HALF_OPEN transition happens once
        the cooldown has elapsed. For HALF_OPEN, only one trial request is
        allowed through at a time -- concurrent callers while a trial is
        already in flight are denied (fall back), not queued as a second
        trial.
        """
        with self._lock:
            s = self._state_for(backend)

            if s.state == CLOSED:
                return True

            if s.state == OPEN:
                elapsed = self._now_fn() - (s.opened_at or 0.0)
                if elapsed >= self.cooldown_seconds:
                    s.state = HALF_OPEN
                    s.half_open_trial_in_flight = True
                    return True
                return False

            if s.state == HALF_OPEN:
                if s.half_open_trial_in_flight:
                    return False
                s.half_open_trial_in_flight = True
                return True

            return False

    def record_success(self, backend: str) -> None:
        with self._lock:
            s = self._state_for(backend)
            if s.state == HALF_OPEN:
                s.state = CLOSED
                s.consecutive_failures = 0
                s.opened_at = None
                s.half_open_trial_in_flight = False
            elif s.state == CLOSED:
                s.consecutive_failures = 0

    def record_failure(self, backend: str) -> None:
        with self._lock:
            s = self._state_for(backend)
            if s.state == HALF_OPEN:
                s.state = OPEN
                s.opened_at = self._now_fn()
                s.half_open_trial_in_flight = False
                s.consecutive_failures = self.failure_threshold  # stays "tripped"
            elif s.state == CLOSED:
                s.consecutive_failures += 1
                if s.consecutive_failures >= self.failure_threshold:
                    s.state = OPEN
                    s.opened_at = self._now_fn()

    def is_available(self, backend: str) -> bool:
        return self.state(backend) != OPEN

    def snapshot(self, backend: str) -> Dict[str, object]:
        with self._lock:
            s = self._state_for(backend)
            return {
                "state": s.state,
                "consecutive_failures": s.consecutive_failures,
                "opened_at": s.opened_at,
                "half_open_trial_in_flight": s.half_open_trial_in_flight,
            }
