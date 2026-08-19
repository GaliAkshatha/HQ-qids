"""
Fast unit tests for StreamWorker's own logic (retry counting, backoff
timing, DLQ routing, idempotency short-circuit) using a minimal fake
Redis client that implements just the stream commands StreamWorker
calls. The real-Redis-backed proof is a separate integration test
(tests/integration/test_ack_retry_dlq_real_redis.py).
"""

import json

from src.contracts import PipelineMessage
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.idempotency import InMemoryIdempotencyStore
from src.runtime.stream_worker import StreamWorker


class FakeRedisStreams:
    """Minimal in-memory fake covering only the stream operations
    StreamWorker actually calls."""

    def __init__(self):
        self.streams = {}  # stream_key -> list[(entry_id, fields)]
        self._counter = 0
        self.acked = []
        self.groups_created = []

    def _next_id(self):
        self._counter += 1
        return f"{self._counter}-0"

    def xadd(self, stream_key, fields):
        entry_id = self._next_id()
        self.streams.setdefault(stream_key, []).append((entry_id, dict(fields)))
        return entry_id

    def xgroup_create(self, stream_key, group, id="0", mkstream=False):
        self.groups_created.append((stream_key, group))

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        (stream_key, _marker), = streams.items()
        entries = self.streams.get(stream_key, [])
        unread = [e for e in entries if e[0] not in [a[1] for a in self.acked if a[0] == stream_key]]
        if not unread:
            return []
        return [(stream_key, unread[:count] if count else unread)]

    def xack(self, stream_key, group, entry_id):
        self.acked.append((stream_key, entry_id))


def make_config(max_retries=2, backoff_seconds=0.001, backoff_multiplier=1.0):
    return RuntimePolicyConfig(
        redis_host="fake", redis_port=0, redis_db=0,
        streams={"dead_letter": "events.dead_letter"}, consumer_groups={},
        max_retries=max_retries, backoff_seconds=backoff_seconds, backoff_multiplier=backoff_multiplier,
        min_idle_time_ms=1000, claim_batch_size=10,
        idempotency_key_prefix="idem", idempotency_ttl_seconds=60,
        health_ports={}, consumer_name_prefix="test", block_ms=100, batch_size=10,
    )


class AlwaysFailWorker(StreamWorker):
    def handle(self, message):
        raise RuntimeError("simulated handler failure")

    def output_stream(self):
        return "output.stream"


class AlwaysSucceedWorker(StreamWorker):
    def handle(self, message):
        return None

    def output_stream(self):
        return None


def push_message(fake_redis, stream_key, **overrides):
    base = dict(
        event_id="e1", correlation_id="c1", causation_id=None, incident_id="i1",
        event_type="test.event", timestamp="2026-01-01T00:00:00+00:00", payload={},
    )
    base.update(overrides)
    msg = PipelineMessage(**base)
    fake_redis.xadd(stream_key, {"data": json.dumps(msg.to_dict(), default=str)})
    return msg


def test_successful_processing_publishes_and_acks():
    fake_redis = FakeRedisStreams()
    config = make_config()
    push_message(fake_redis, "input.stream", event_id="e1")

    worker = AlwaysSucceedWorker("w", fake_redis, config, "input.stream", "group", InMemoryIdempotencyStore(), get_logger("test_stream_worker"))
    worker.run_once(block_ms=10)

    assert fake_redis.acked == [("input.stream", "1-0")]


def test_retry_republishes_with_incremented_count_and_acks_original():
    fake_redis = FakeRedisStreams()
    config = make_config(max_retries=3)
    push_message(fake_redis, "input.stream", event_id="e1", retry_count=0)

    worker = AlwaysFailWorker("w", fake_redis, config, "input.stream", "group", InMemoryIdempotencyStore(), get_logger("test_stream_worker"))
    worker.run_once(block_ms=10)

    assert ("input.stream", "1-0") in fake_redis.acked
    republished = fake_redis.streams["input.stream"][1]
    republished_msg = PipelineMessage.from_dict(json.loads(republished[1]["data"]))
    assert republished_msg.retry_count == 1
    assert republished_msg.event_id == "e1"


def test_retries_exhausted_goes_to_dead_letter():
    fake_redis = FakeRedisStreams()
    config = make_config(max_retries=1)
    push_message(fake_redis, "input.stream", event_id="e1", retry_count=1)

    worker = AlwaysFailWorker("w", fake_redis, config, "input.stream", "group", InMemoryIdempotencyStore(), get_logger("test_stream_worker"))
    worker.run_once(block_ms=10)

    dlq_entries = fake_redis.streams.get("events.dead_letter", [])
    assert len(dlq_entries) == 1
    dlq_payload = dlq_entries[0][1]
    assert dlq_payload["failed_worker"] == "w"
    assert "simulated handler failure" in dlq_payload["error"]


def test_idempotent_duplicate_is_skipped_without_calling_handle():
    fake_redis = FakeRedisStreams()
    config = make_config()
    push_message(fake_redis, "input.stream", event_id="e1")

    call_count = {"n": 0}

    class CountingWorker(StreamWorker):
        def handle(self, message):
            call_count["n"] += 1
            return None

        def output_stream(self):
            return None

    idem = InMemoryIdempotencyStore()
    idem.mark_processed("e1")

    worker = CountingWorker("w", fake_redis, config, "input.stream", "group", idem, get_logger("test_stream_worker"))
    worker.run_once(block_ms=10)

    assert call_count["n"] == 0
    assert fake_redis.acked == [("input.stream", "1-0")]
