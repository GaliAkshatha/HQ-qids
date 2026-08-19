"""
src/incident/event_store.py

EventStore is deliberately narrow (append, read_all, event_exists) so
Phase 7 can swap in a distributed implementation (Redis Streams,
RabbitMQ, etc.) behind the same interface without IncidentManager
changing at all.

JsonlEventStore is NOT just a log -- reading it back (read_all) is how a
brand-new process reconstructs incident state after a restart. This is
what makes it "meaningful incident state persistence" rather than mere
logging, per the explicit requirement: on construction, the entire file
is read into memory so both event-level dedup (event_id) and
read_all(correlation_id) work correctly even for a process that never
called append() itself.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set

from src.contracts import IncidentEvent


class EventStore(ABC):
    @abstractmethod
    def append(self, event: IncidentEvent) -> bool:
        """Appends an event. Returns True if newly appended, False if
        event_id was already seen (event-level idempotency -- Decision 5A).
        Never appends the same event_id twice."""
        raise NotImplementedError

    @abstractmethod
    def read_all(self, correlation_id: Optional[str] = None) -> List[IncidentEvent]:
        """Returns events in append order, optionally filtered by
        correlation_id. This is the replay source for state reconstruction."""
        raise NotImplementedError

    @abstractmethod
    def event_exists(self, event_id: str) -> bool:
        raise NotImplementedError


class InMemoryEventStore(EventStore):
    """No persistence across process restarts -- for tests and any caller
    that doesn't need durability. Same dedup/ordering semantics as
    JsonlEventStore so tests written against one behave identically
    against the other."""

    def __init__(self) -> None:
        self._events: List[IncidentEvent] = []
        self._seen_ids: Set[str] = set()
        self._lock = threading.Lock()

    def append(self, event: IncidentEvent) -> bool:
        with self._lock:
            if event.event_id in self._seen_ids:
                return False
            self._seen_ids.add(event.event_id)
            self._events.append(event)
            return True

    def read_all(self, correlation_id: Optional[str] = None) -> List[IncidentEvent]:
        with self._lock:
            if correlation_id is None:
                return list(self._events)
            return [e for e in self._events if e.correlation_id == correlation_id]

    def event_exists(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._seen_ids


class JsonlEventStore(EventStore):
    """
    Append-only JSON-lines file. On construction, reads the entire file
    (if it exists) to build the in-memory index -- this is what lets a
    freshly-constructed instance (e.g. after a "restart") correctly
    report event_exists()/read_all() for events it did not itself append,
    proving the file is the actual source of truth, not just a log next
    to a separate authoritative in-memory store.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: List[IncidentEvent] = []
        self._seen_ids: Set[str] = set()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                event = IncidentEvent.from_dict(data)
                if event.event_id not in self._seen_ids:
                    self._seen_ids.add(event.event_id)
                    self._events.append(event)

    def append(self, event: IncidentEvent) -> bool:
        with self._lock:
            if event.event_id in self._seen_ids:
                return False
            with open(self.path, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
            self._seen_ids.add(event.event_id)
            self._events.append(event)
            return True

    def read_all(self, correlation_id: Optional[str] = None) -> List[IncidentEvent]:
        with self._lock:
            if correlation_id is None:
                return list(self._events)
            return [e for e in self._events if e.correlation_id == correlation_id]

    def event_exists(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._seen_ids
