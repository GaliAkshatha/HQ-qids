"""
Tests SSE event delivery by reading a bounded number of frames from the
real stream generator against a real experiment_service with real
persisted events -- not a mock.
"""

from src.api.app import create_app
from src.api.services.event_service import stream_incident_events


def test_stream_incident_events_yields_real_events():
    app = create_app()
    service = app.config["EXPERIMENT_SERVICE"]

    experiment = service.create_experiment(scenario="normal_browsing", n_sessions=1, mode="normal", quantum="auto")
    service.run_experiment(experiment.experiment_id)

    gen = stream_incident_events(service, poll_interval=0.01, max_iterations=2)
    frames = list(gen)

    assert len(frames) > 0
    joined = "".join(frames)
    assert "event:" in joined
    assert "data:" in joined
    assert "DETECTION_COMPLETED" in joined or "INCIDENT_UPDATED" in joined


def test_stream_format_is_valid_sse():
    from src.api.services.event_service import format_sse

    frame = format_sse({"a": 1}, event="TEST_EVENT")
    assert frame.startswith("event: TEST_EVENT\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
