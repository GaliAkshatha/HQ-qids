import json
import uuid

import pytest
import redis

from src.agents.application_feedback import get_feedback_for_incident
from src.agents.feedback_listener import FeedbackListener
from src.contracts import PipelineMessage
from src.runtime.config import RuntimePolicyConfig


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_config():
    import dataclasses
    base = RuntimePolicyConfig.load()
    suffix = uuid.uuid4().hex[:8]
    streams = {k: f"{v}.appfb-test-{suffix}" for k, v in base.streams.items()}
    return dataclasses.replace(base, streams=streams)


def test_application_feedback_connection_point_reuses_real_feedback_listener(redis_client):
    """Proves the connection mechanism works via real Redis, using a
    synthetic incident_id -- exactly the same pattern Phase 8's own
    FeedbackListener tests use. Does NOT claim a real Suzume-driven
    incident exists yet (see docs/APPLICATION_SECURITY_MODEL_BOUNDARY.md)."""
    config = make_config()
    listener = FeedbackListener(redis_client, config)

    synthetic_incident_id = "synthetic-app-incident-1"
    msg = PipelineMessage(
        event_id=str(uuid.uuid4()), correlation_id="sess-1", causation_id=None,
        incident_id=synthetic_incident_id, event_type="test.event", timestamp="2026-01-01T00:00:00+00:00",
        payload={"current_state": "RESOLVED", "escalated": False},
    )
    redis_client.xadd(config.streams["incident_updated"], {"data": json.dumps(msg.to_dict(), default=str)})

    outcome = get_feedback_for_incident(synthetic_incident_id, listener)
    assert outcome is not None
    assert outcome.incident_current_state == "RESOLVED"


def test_application_feedback_returns_none_for_unknown_incident(redis_client):
    config = make_config()
    listener = FeedbackListener(redis_client, config)
    assert get_feedback_for_incident("never-existed", listener) is None


def test_application_feedback_module_is_read_only():
    import src.agents.application_feedback as mod
    source = open(mod.__file__).read()
    assert "xadd(" not in source
    assert "xack(" not in source
