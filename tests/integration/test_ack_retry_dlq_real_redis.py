"""
Real Redis proof of ACK/retry/DLQ behavior -- not mocked. Each test uses
a uniquely-named stream/group so tests don't interfere with each other
or with anything else using this Redis instance.
"""

import json
import time
import uuid

import pytest
import redis

from src.contracts import PipelineMessage
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.idempotency import InMemoryIdempotencyStore
from src.runtime.stream_worker import StreamWorker


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_config(max_retries=2, backoff_seconds=0.01, min_idle_time_ms=200):
    return RuntimePolicyConfig(
        redis_host="localhost", redis_port=6379, redis_db=0,
        streams={"dead_letter": f"test.dlq.{uuid.uuid4().hex[:8]}"}, consumer_groups={},
        max_retries=max_retries, backoff_seconds=backoff_seconds, backoff_multiplier=1.0,
        min_idle_time_ms=min_idle_time_ms, claim_batch_size=10,
        idempotency_key_prefix=f"test_idem_{uuid.uuid4().hex[:8]}", idempotency_ttl_seconds=60,
        health_ports={}, consumer_name_prefix="test", block_ms=100, batch_size=10,
    )


def push(redis_client, stream, **overrides):
    base = dict(
        event_id=str(uuid.uuid4()), correlation_id="c1", causation_id=None, incident_id="i1",
        event_type="test.event", timestamp="2026-01-01T00:00:00+00:00", payload={},
    )
    base.update(overrides)
    msg = PipelineMessage(**base)
    redis_client.xadd(stream, {"data": json.dumps(msg.to_dict(), default=str)})
    return msg


def test_real_redis_successful_ack(redis_client):
    stream = f"test.ack.{uuid.uuid4().hex[:8]}"
    group = f"{stream}.group"
    config = make_config()
    push(redis_client, stream, event_id="e1")

    class OkWorker(StreamWorker):
        def handle(self, message):
            return None

        def output_stream(self):
            return None

    worker = OkWorker("w", redis_client, config, stream, group, InMemoryIdempotencyStore(), get_logger("test_ack"))
    worker.run_once(block_ms=500)

    pending = redis_client.xpending(stream, group)
    assert pending["pending"] == 0


def test_real_redis_retry_then_dlq(redis_client):
    stream = f"test.retry.{uuid.uuid4().hex[:8]}"
    group = f"{stream}.group"
    config = make_config(max_retries=2, backoff_seconds=0.01)
    push(redis_client, stream, event_id="e1")

    class AlwaysFailWorker(StreamWorker):
        def handle(self, message):
            raise RuntimeError("intentional failure")

        def output_stream(self):
            return "unused"

    worker = AlwaysFailWorker("w", redis_client, config, stream, group, InMemoryIdempotencyStore(), get_logger("test_retry"))

    worker.run_once(block_ms=500)
    worker.run_once(block_ms=500)
    worker.run_once(block_ms=500)

    dlq_entries = redis_client.xrange(config.streams["dead_letter"], min="-", max="+")
    assert len(dlq_entries) == 1
    dlq_data = dlq_entries[0][1]
    assert dlq_data["failed_worker"] == "w"
    assert "intentional failure" in dlq_data["error"]

    pending = redis_client.xpending(stream, group)
    assert pending["pending"] == 0


def test_real_redis_xautoclaim_recovers_pending_message_after_simulated_crash(redis_client):
    """Simulates a worker crash: read the message (creating a pending
    entry) but never ACK or handle it -- then a second worker instance
    uses XAUTOCLAIM to recover and successfully process it."""
    stream = f"test.crash.{uuid.uuid4().hex[:8]}"
    group = f"{stream}.group"
    config = make_config(min_idle_time_ms=100)
    push(redis_client, stream, event_id="e1")

    redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    redis_client.xreadgroup(group, "crashed-consumer", {stream: ">"}, count=10)

    pending_before = redis_client.xpending(stream, group)
    assert pending_before["pending"] == 1

    time.sleep(0.15)

    processed = []

    class RecoveringWorker(StreamWorker):
        def handle(self, message):
            processed.append(message.event_id)
            return None

        def output_stream(self):
            return None

    recovering_worker = RecoveringWorker("w2", redis_client, config, stream, group, InMemoryIdempotencyStore(), get_logger("test_crash"))
    claimed = recovering_worker.recover_pending()

    assert len(claimed) == 1
    assert processed == ["e1"]

    pending_after = redis_client.xpending(stream, group)
    assert pending_after["pending"] == 0
