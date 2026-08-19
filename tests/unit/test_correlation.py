from src.contracts import IncidentEvent
from src.contracts.incident import DETECTED, DETECTION_CREATED
from src.incident.correlation import (
    CorrelationStrategy, RepeatedIncidentTracker, SampleIdCorrelation, build_tracker_from_events,
)


def test_sample_id_correlation_returns_sample_id_unchanged():
    strategy = SampleIdCorrelation()
    assert strategy.correlation_key("sample-42") == "sample-42"


def test_correlation_strategy_is_replaceable_through_the_interface():
    """Proves IncidentManager could be handed a different strategy (e.g. a
    future real network identity) without any change to how it's called."""

    class FakeSourceIpCorrelation(CorrelationStrategy):
        def correlation_key(self, sample_id: str, **kwargs) -> str:
            return kwargs.get("source_ip", "unknown")

    strategy: CorrelationStrategy = FakeSourceIpCorrelation()
    assert strategy.correlation_key("sample-1", source_ip="10.0.0.5") == "10.0.0.5"
    # same interface call shape as SampleIdCorrelation, different result
    assert SampleIdCorrelation().correlation_key("sample-1", source_ip="10.0.0.5") == "sample-1"


def test_repeated_incident_tracker_counts_correctly():
    tracker = RepeatedIncidentTracker()
    assert tracker.record_incident("target-a") == 1
    assert tracker.record_incident("target-a") == 2
    assert tracker.record_incident("target-b") == 1
    assert tracker.record_incident("target-a") == 3
    assert tracker.count_for("target-a") == 3
    assert tracker.count_for("target-b") == 1
    assert tracker.count_for("never-seen") == 0


def make_detection_event(event_id, incident_id, correlation_id):
    return IncidentEvent(
        event_id=event_id, correlation_id=correlation_id, incident_id=incident_id, event_type=DETECTION_CREATED,
        previous_state=DETECTED, new_state=DETECTED, timestamp="2026-01-01T00:00:00+00:00", reason="ok",
    )


def test_build_tracker_from_events_counts_distinct_incidents_per_key():
    events = [
        make_detection_event("e1", "inc-1", "target-a"),
        make_detection_event("e2", "inc-2", "target-a"),  # different incident, SAME correlation key
        make_detection_event("e3", "inc-3", "target-b"),
    ]
    tracker = build_tracker_from_events(events)
    assert tracker.count_for("target-a") == 2
    assert tracker.count_for("target-b") == 1


def test_build_tracker_does_not_double_count_multiple_events_from_same_incident():
    events = [
        make_detection_event("e1", "inc-1", "target-a"),
        # a hypothetical duplicate/replayed DETECTION_CREATED for the SAME incident_id
        make_detection_event("e2", "inc-1", "target-a"),
    ]
    tracker = build_tracker_from_events(events)
    assert tracker.count_for("target-a") == 1
