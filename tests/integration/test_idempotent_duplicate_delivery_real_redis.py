import json
import uuid

import pytest
import redis

from src.contracts import PipelineMessage
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.stream_worker import StreamWorker


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_config():
    return RuntimePolicyConfig(
        redis_host="localhost", redis_port=6379, redis_db=0,
        streams={"dead_letter": f"test.dlq.{uuid.uuid4().hex[:8]}"}, consumer_groups={},
        max_retries=2, backoff_seconds=0.01, backoff_multiplier=1.0,
        min_idle_time_ms=200, claim_batch_size=10,
        idempotency_key_prefix=f"test_idem_{uuid.uuid4().hex[:8]}", idempotency_ttl_seconds=60,
        health_ports={}, consumer_name_prefix="test", block_ms=100, batch_size=10,
    )


def test_real_redis_duplicate_delivery_is_detected_and_side_effect_runs_once(redis_client):
    """1. event processed once. 2. same event delivered again.
    3. second delivery detected. 4/5. no duplicate side effect
    (simulated via a call counter standing in for a defense action /
    incident transition). 6. event safely acknowledged both times."""
    stream = f"test.dup.{uuid.uuid4().hex[:8]}"
    group = f"{stream}.group"
    config = make_config()
    idem = RedisIdempotencyStore(redis_client, config, "dup-test-worker")

    event_id = str(uuid.uuid4())
    msg = PipelineMessage(
        event_id=event_id, correlation_id="c1", causation_id=None, incident_id="i1",
        event_type="test.event", timestamp="2026-01-01T00:00:00+00:00", payload={},
    )

    side_effect_calls = []

    class SideEffectWorker(StreamWorker):
        def handle(self, message):
            side_effect_calls.append(message.event_id)
            return None

        def output_stream(self):
            return None

    worker = SideEffectWorker("w", redis_client, config, stream, group, idem, get_logger("test_dup"))

    # 1. deliver once, process
    redis_client.xadd(stream, {"data": json.dumps(msg.to_dict(), default=str)})
    worker.run_once(block_ms=500)
    assert side_effect_calls == [event_id]

    # 2. same event_id "delivered again"
    redis_client.xadd(stream, {"data": json.dumps(msg.to_dict(), default=str)})
    worker.run_once(block_ms=500)

    # 3/4/5. detected as duplicate, side effect NOT invoked a second time
    assert side_effect_calls == [event_id]

    # 6. both deliveries were safely acknowledged
    pending = redis_client.xpending(stream, group)
    assert pending["pending"] == 0
