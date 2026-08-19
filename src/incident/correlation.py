"""
src/incident/correlation.py

NSL-KDD provides no real source-IP/session identity. sample_id is used
as the correlation key today -- this is a simulation correlation
strategy, not a claim of real network attribution. CorrelationStrategy
exists specifically so a future real identity (source_ip, session_id,
flow_id) can replace SampleIdCorrelation without IncidentManager itself
changing at all -- it only ever calls `strategy.correlation_key(...)`.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Dict, List

from src.contracts import IncidentEvent
from src.contracts.incident import DETECTION_CREATED


class CorrelationStrategy(ABC):
    @abstractmethod
    def correlation_key(self, sample_id: str, **kwargs) -> str:
        raise NotImplementedError


class SampleIdCorrelation(CorrelationStrategy):
    """
    Current (Phase 6) strategy: sample_id IS the correlation key.
    Every NSL-KDD row has a unique sample_id, so in practice each
    incident is its own, unrelated correlation group -- "repeated
    incidents for the same target" is real, tested code, but will not
    organically occur against this dataset. A future strategy grouping
    by a real network identity would let multiple distinct incidents
    (different DetectionResults, different incident_ids) share one
    correlation key, which is exactly what RepeatedIncidentTracker below
    is built to count correctly once that's true.
    """

    def correlation_key(self, sample_id: str, **kwargs) -> str:
        return sample_id


class RepeatedIncidentTracker:
    """Counts DISTINCT incidents (not events) created per correlation
    key, so escalation.py can check 'has this target had N prior
    incidents'. Restart-safe via build_tracker_from_events()."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record_incident(self, correlation_key: str) -> int:
        """Records a newly created incident for this key. Returns the
        total count for this key, including this one."""
        with self._lock:
            self._counts[correlation_key] = self._counts.get(correlation_key, 0) + 1
            return self._counts[correlation_key]

    def count_for(self, correlation_key: str) -> int:
        with self._lock:
            return self._counts.get(correlation_key, 0)


def build_tracker_from_events(events: List[IncidentEvent]) -> RepeatedIncidentTracker:
    """
    Reconstructs a RepeatedIncidentTracker from persisted event history
    -- counts distinct incident_ids per correlation_id among
    DETECTION_CREATED events (one per incident, by construction), so a
    restarted process's repeated-incident counting is correct from the
    very first new incident it processes, not reset to zero.
    """
    tracker = RepeatedIncidentTracker()
    seen_incident_ids_per_key: Dict[str, set] = {}
    for event in events:
        if event.event_type != DETECTION_CREATED:
            continue
        key = event.correlation_id
        seen_incident_ids_per_key.setdefault(key, set())
        if event.incident_id not in seen_incident_ids_per_key[key]:
            seen_incident_ids_per_key[key].add(event.incident_id)
            tracker.record_incident(key)
    return tracker
