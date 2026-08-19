"""
Uses the real local redis-server (available in this environment) rather
than mocking Redis for the event store -- these are effectively fast
integration tests, kept in unit/ because they run in milliseconds and
need no multi-service setup, unlike the heavier distributed pipeline
tests.
"""

import uuid

import pytest
import redis

from src.contracts import IncidentEvent
from src.contracts.incident import DETECTED, DETECTION_CREATED
from src.runtime.event_store_redis import RedisEventStore


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_event(event_id, correlation_id="s1", incident_id="inc-1", previous=DETECTED, new=DETECTED):
    return IncidentEvent(
        event_id=event_id, correlation_id=correlation_id, incident_id=incident_id, event_type=DETECTION_CREATED,
        previous_state=previous, new_state=new, timestamp="2026-01-01T00:00:00+00:00", reason="ok",
    )


def test_append_and_read_all(redis_client):
    prefix = f"test_event_store_{uuid.uuid4().hex[:8]}"
    store = RedisEventStore(redis_client, key_prefix=prefix)
    store.append(make_event("e1"))
    store.append(make_event("e2"))

    events = store.read_all()
    assert [e.event_id for e in events] == ["e1", "e2"]


def test_duplicate_event_id_not_appended_twice(redis_client):
    prefix = f"test_event_store_{uuid.uuid4().hex[:8]}"
    store = RedisEventStore(redis_client, key_prefix=prefix)
    first = store.append(make_event("e1"))
    second = store.append(make_event("e1"))
    assert first is True
    assert second is False
    assert len(store.read_all()) == 1


def test_read_all_filters_by_correlation_id(redis_client):
    prefix = f"test_event_store_{uuid.uuid4().hex[:8]}"
    store = RedisEventStore(redis_client, key_prefix=prefix)
    store.append(make_event("e1", correlation_id="s1"))
    store.append(make_event("e2", correlation_id="s2"))
    assert [e.event_id for e in store.read_all("s1")] == ["e1"]


def test_event_exists(redis_client):
    prefix = f"test_event_store_{uuid.uuid4().hex[:8]}"
    store = RedisEventStore(redis_client, key_prefix=prefix)
    store.append(make_event("e1"))
    assert store.event_exists("e1") is True
    assert store.event_exists("e2") is False


def test_new_instance_same_prefix_sees_prior_events(redis_client):
    """Same durability property JsonlEventStore has across a restart --
    here proven across two separate RedisEventStore instances."""
    prefix = f"test_event_store_{uuid.uuid4().hex[:8]}"
    store_1 = RedisEventStore(redis_client, key_prefix=prefix)
    store_1.append(make_event("e1"))

    store_2 = RedisEventStore(redis_client, key_prefix=prefix)
    assert store_2.event_exists("e1") is True
    assert [e.event_id for e in store_2.read_all()] == ["e1"]
