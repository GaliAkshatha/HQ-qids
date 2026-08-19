import pytest

from src.contracts import PipelineMessage


def make_message(**overrides):
    base = dict(
        event_id="e1", correlation_id="c1", causation_id=None, incident_id="i1",
        event_type="traffic.ingested", timestamp="2026-01-01T00:00:00+00:00", payload={"a": 1},
    )
    base.update(overrides)
    return PipelineMessage(**base)


def test_serialization_round_trip():
    msg = make_message(causation_id="parent-1", retry_count=2)
    data = msg.to_dict()
    reloaded = PipelineMessage.from_dict(data)
    assert reloaded == msg


def test_requires_event_id():
    with pytest.raises(ValueError):
        make_message(event_id="")


def test_requires_correlation_id():
    with pytest.raises(ValueError):
        make_message(correlation_id="")


def test_requires_incident_id():
    with pytest.raises(ValueError):
        make_message(incident_id="")


def test_requires_event_type():
    with pytest.raises(ValueError):
        make_message(event_type="")


def test_negative_retry_count_rejected():
    with pytest.raises(ValueError):
        make_message(retry_count=-1)


def test_causation_id_none_allowed_for_root_event():
    msg = make_message(causation_id=None)
    assert msg.causation_id is None


def test_next_retry_increments_without_mutating_original():
    original = make_message(retry_count=0)
    retried = original.next_retry()
    assert retried.retry_count == 1
    assert original.retry_count == 0  # original untouched
    assert retried.event_id == original.event_id  # same logical event, just retried
