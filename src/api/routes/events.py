from flask import Blueprint, Response, current_app, jsonify, stream_with_context

from src.api.services.event_service import stream_incident_events

events_bp = Blueprint("events", __name__)


@events_bp.get("/events")
def event_stream():
    service = current_app.config["EXPERIMENT_SERVICE"]
    return Response(
        stream_with_context(stream_incident_events(service)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@events_bp.get("/events/<correlation_id>")
def events_for_correlation(correlation_id):
    service = current_app.config["EXPERIMENT_SERVICE"]
    all_events = service._event_store.read_all(correlation_id)
    return jsonify({
        "correlation_id": correlation_id,
        "events": [
            {"event_type": e.event_type, "incident_id": e.incident_id, "reason": e.reason, "timestamp": e.timestamp}
            for e in all_events
        ],
    })
