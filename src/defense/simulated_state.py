"""
src/defense/simulated_state.py

The simulated environment every defense action actually operates on.
NOT a mock that returns "success" -- this holds real, mutable,
independently-inspectable state per target, so verification (Phase 5's
independent check) can read it fresh rather than trusting whatever the
executor claims happened.

NSL-KDD provides no source-IP or session identity, so `target` here is
the sample_id itself -- each classified sample is treated as its own
isolated simulated source, not grouped by a real network identity. This
is a real limitation of the underlying dataset, not something Phase 5
works around; documented here so it isn't implied to be more than it is.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Set


@dataclass
class SourceState:
    target: str
    blocked: bool = False
    isolated: bool = False
    rate_limited: bool = False
    monitored: bool = False
    monitoring_level: str = "BASELINE"  # "BASELINE" | "INCREASED"
    active_sessions: int = 1
    suspended_sessions: int = 0
    terminated: bool = False


class SimulatedNetworkState:
    """
    In-memory state for all simulated targets. Every executor action
    mutates a SourceState here; verification reads it back independently.
    """

    def __init__(self) -> None:
        self._sources: Dict[str, SourceState] = {}
        # explicit, isolated test-only fault injection -- never touched by
        # production code paths, only by tests that opt in
        self._inject_failure_for: Set[str] = set()

    def get_or_create(self, target: str) -> SourceState:
        if target not in self._sources:
            self._sources[target] = SourceState(target=target)
        return self._sources[target]

    def get(self, target: str) -> Optional[SourceState]:
        return self._sources.get(target)

    def snapshot(self, target: str) -> SourceState:
        """A deep copy of the target's current state -- used as the
        pre-action baseline that rollback restores to exactly."""
        return copy.deepcopy(self.get_or_create(target))

    def restore(self, target: str, snapshot: SourceState) -> None:
        """Restore the COMPLETE state for a target from a prior snapshot
        -- every field, not just the one the triggering action touched."""
        self._sources[target] = copy.deepcopy(snapshot)

    # ---- test-only fault injection (isolated, never used in production paths) ----

    def inject_failure_for(self, target: str) -> None:
        self._inject_failure_for.add(target)

    def clear_injected_failure(self, target: str) -> None:
        self._inject_failure_for.discard(target)

    def should_fail(self, target: str) -> bool:
        return target in self._inject_failure_for
