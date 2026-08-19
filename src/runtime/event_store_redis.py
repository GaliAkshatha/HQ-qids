"""
src/runtime/event_store_redis.py

RedisEventStore implements Phase 6's EventStore ABC exactly --
InMemoryEventStore and JsonlEventStore remain fully intact and are NOT
replaced. This is an alternative backing store IncidentManager could use
instead of JsonlEventStore, for durability that survives a process
restart the same way JsonlEventStore does, but backed by Redis instead of
a local file.

Deliberately distinct from the Redis Streams messaging layer
(traffic.ingested, detection.completed, etc.) used by src/runtime's
workers -- those carry PipelineMessage objects for inter-service
orchestration; this stores IncidentEvent objects for incident audit
history. Different schema, different purpose, kept separate on purpose.
"""

from __future__ import annotations

import json
from typing import List, Optional

import redis

from src.contracts import IncidentEvent
from src.incident.event_store import EventStore


class RedisEventStore(EventStore):
    def __init__(self, client: redis.Redis, key_prefix: str = "incident_event_store") -> None:
        self.client = client
        self._stream_key = f"{key_prefix}:stream"
        self._seen_ids_key = f"{key_prefix}:seen_ids"

    def append(self, event: IncidentEvent) -> bool:
        if self.client.sismember(self._seen_ids_key, event.event_id):
            return False
        self.client.xadd(self._stream_key, {"data": json.dumps(event.to_dict(), default=str)})
        self.client.sadd(self._seen_ids_key, event.event_id)
        return True

    def read_all(self, correlation_id: Optional[str] = None) -> List[IncidentEvent]:
        entries = self.client.xrange(self._stream_key, min="-", max="+")
        events = [IncidentEvent.from_dict(json.loads(fields["data"])) for _, fields in entries]
        if correlation_id is None:
            return events
        return [e for e in events if e.correlation_id == correlation_id]

    def event_exists(self, event_id: str) -> bool:
        return bool(self.client.sismember(self._seen_ids_key, event_id))
