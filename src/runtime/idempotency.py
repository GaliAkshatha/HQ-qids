"""
src/runtime/idempotency.py

Explicit event-level idempotency, NOT relying on Redis consumer groups
alone (consumer groups guarantee delivery, not exactly-once processing --
a redelivered message after a crash-before-ACK is a legitimate, expected
occurrence, not a bug). This is the mechanism that makes a redelivered
event a safe no-op for the worker's own side effects.

Ordering contract (documented, not just implied): a worker must call
mark_processed() only AFTER its side effect (and any republish) has
already succeeded, immediately before ACKing. This is at-least-once
delivery with idempotent effects, not exactly-once -- a crash between
"side effect succeeded" and "mark_processed" would cause reprocessing on
redelivery, which is safe precisely because the side effect itself is
idempotent (Phase 5's already-active check, Phase 6's terminal-incident
check). Documented plainly rather than overclaiming exactly-once.
"""

from __future__ import annotations

import redis

from src.runtime.config import RuntimePolicyConfig


class RedisIdempotencyStore:
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, worker_name: str) -> None:
        self.client = client
        self.ttl_seconds = config.idempotency_ttl_seconds
        self.key_prefix = config.idempotency_key_prefix
        self.worker_name = worker_name

    def _key(self, event_id: str) -> str:
        return f"{self.key_prefix}:{self.worker_name}:{event_id}"

    def already_processed(self, event_id: str) -> bool:
        return bool(self.client.exists(self._key(event_id)))

    def mark_processed(self, event_id: str) -> None:
        self.client.set(self._key(event_id), "1", ex=self.ttl_seconds)


class InMemoryIdempotencyStore:
    """Fast, no-Redis-required alternative for unit tests -- same
    interface as RedisIdempotencyStore."""

    def __init__(self) -> None:
        self._seen = set()

    def already_processed(self, event_id: str) -> bool:
        return event_id in self._seen

    def mark_processed(self, event_id: str) -> None:
        self._seen.add(event_id)
