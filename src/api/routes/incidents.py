from flask import Blueprint, current_app, jsonify

incidents_bp = Blueprint("incidents", __name__)


@incidents_bp.get("/incidents")
def list_incidents():
    service = current_app.config["EXPERIMENT_SERVICE"]
    snapshots = service.list_incidents()
    return jsonify({"incidents": [s.to_dict() for s in snapshots]})


@incidents_bp.get("/incidents/<incident_id>")
def get_incident(incident_id):
    service = current_app.config["EXPERIMENT_SERVICE"]
    snapshot = service.get_incident(incident_id)
    if snapshot is None:
        return jsonify({"error": "not found"}), 404
    events = service.get_events(incident_id)
    return jsonify({
        "incident": snapshot.to_dict(),
        "timeline": [
            {
                "event_type": e.event_type, "previous_state": e.previous_state, "new_state": e.new_state,
                "reason": e.reason, "timestamp": e.timestamp, "payload": e.payload,
            }
            for e in events
        ],
    })
