import json
import uuid

import pytest
import redis

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
    streams = {k: f"{v}.fb-test-{suffix}" for k, v in base.streams.items()}
    return dataclasses.replace(base, streams=streams)


def publish(redis_client, stream, correlation_id, incident_id, payload):
    msg = PipelineMessage(
        event_id=str(uuid.uuid4()), correlation_id=correlation_id, causation_id=None,
        incident_id=incident_id, event_type="test.event", timestamp="2026-01-01T00:00:00+00:00", payload=payload,
    )
    redis_client.xadd(stream, {"data": json.dumps(msg.to_dict(), default=str)})


def test_collect_outcome_filters_by_incident_id_not_correlation_id(redis_client):
    config = make_config()
    listener = FeedbackListener(redis_client, config)

    session_id = "shared-session"
    publish(redis_client, config.streams["incident_updated"], session_id, "inc-A", {"current_state": "RESOLVED", "escalated": False})
    publish(redis_client, config.streams["incident_updated"], session_id, "inc-B", {"current_state": "ESCALATED", "escalated": True})

    outcome_a = listener.collect_outcome("inc-A")
    outcome_b = listener.collect_outcome("inc-B")

    assert outcome_a.incident_current_state == "RESOLVED"
    assert outcome_a.escalated is False
    assert outcome_b.incident_current_state == "ESCALATED"
    assert outcome_b.escalated is True


def test_collect_outcome_merges_defense_completed_data(redis_client):
    config = make_config()
    listener = FeedbackListener(redis_client, config)

    publish(redis_client, config.streams["defense_completed"], "sess1", "inc-1", {
        "defense_result": {"action": "ISOLATE_SIMULATED_SOURCE", "severity": "HIGH"},
        "hybrid_decision": {"final_prediction": "attack"},
    })
    publish(redis_client, config.streams["incident_updated"], "sess1", "inc-1", {"current_state": "RESOLVED", "escalated": False})

    outcome = listener.collect_outcome("inc-1")
    assert outcome.selected_action == "ISOLATE_SIMULATED_SOURCE"
    assert outcome.risk_level == "HIGH"
    assert outcome.final_prediction == "attack"


def test_collect_outcome_returns_none_for_unknown_incident(redis_client):
    config = make_config()
    listener = FeedbackListener(redis_client, config)
    assert listener.collect_outcome("never-published") is None


def test_session_escalation_count_counts_distinct_escalated_incidents(redis_client):
    config = make_config()
    listener = FeedbackListener(redis_client, config)

    session_id = "repeat-session"
    publish(redis_client, config.streams["incident_updated"], session_id, "inc-1", {"current_state": "RESOLVED", "escalated": False})
    publish(redis_client, config.streams["incident_updated"], session_id, "inc-2", {"current_state": "ESCALATED", "escalated": True})
    publish(redis_client, config.streams["incident_updated"], session_id, "inc-3", {"current_state": "ESCALATED", "escalated": True})

    assert listener.session_escalation_count(session_id) == 2


def test_feedback_listener_never_acks_or_writes(redis_client):
    import src.agents.feedback_listener as mod
    source = open(mod.__file__).read()
    assert "xack(" not in source
    assert "xreadgroup(" not in source
    assert "xgroup_create(" not in source
    assert "xadd(" not in source
