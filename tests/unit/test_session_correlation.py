import pytest

from src.agents.session_correlation import AgentSessionCorrelation, SESSION_DELIMITER


def test_parses_session_id_from_sample_id():
    strategy = AgentSessionCorrelation()
    key = strategy.correlation_key(f"sess-abc123{SESSION_DELIMITER}0{SESSION_DELIMITER}f9e1c2")
    assert key == "sess-abc123"


def test_multiple_turns_same_session_share_correlation_key():
    strategy = AgentSessionCorrelation()
    key1 = strategy.correlation_key(f"sess-xyz{SESSION_DELIMITER}0{SESSION_DELIMITER}aaa")
    key2 = strategy.correlation_key(f"sess-xyz{SESSION_DELIMITER}1{SESSION_DELIMITER}bbb")
    key3 = strategy.correlation_key(f"sess-xyz{SESSION_DELIMITER}2{SESSION_DELIMITER}ccc")
    assert key1 == key2 == key3 == "sess-xyz"


def test_rejects_sample_id_without_delimiter():
    strategy = AgentSessionCorrelation()
    with pytest.raises(ValueError):
        strategy.correlation_key("no-delimiter-here")


def test_rejects_empty_session_id():
    strategy = AgentSessionCorrelation()
    with pytest.raises(ValueError):
        strategy.correlation_key(f"{SESSION_DELIMITER}0{SESSION_DELIMITER}aaa")


def test_traffic_gateway_default_behavior_unchanged_when_agents_unused():
    """Confirms the existing pipeline still works exactly as before when
    Phase 8 is completely unused -- the default constructor path."""
    from unittest.mock import MagicMock

    from src.incident.correlation import SampleIdCorrelation
    from src.runtime.services.traffic_gateway import TrafficGateway

    fake_client = MagicMock()
    fake_config = MagicMock()
    fake_config.streams = {"traffic_ingested": "traffic.ingested"}

    gateway = TrafficGateway(fake_client, fake_config)
    assert isinstance(gateway.correlation_strategy, SampleIdCorrelation)

    message = gateway.ingest("plain-sample-id", {"a": 1})
    assert message.correlation_id == "plain-sample-id"


def test_traffic_gateway_accepts_agent_session_correlation():
    from unittest.mock import MagicMock

    from src.runtime.services.traffic_gateway import TrafficGateway

    fake_client = MagicMock()
    fake_config = MagicMock()
    fake_config.streams = {"traffic_ingested": "traffic.ingested"}

    gateway = TrafficGateway(fake_client, fake_config, correlation_strategy=AgentSessionCorrelation())
    message = gateway.ingest(f"sess-1{SESSION_DELIMITER}0{SESSION_DELIMITER}aaa", {"a": 1})
    assert message.correlation_id == "sess-1"
