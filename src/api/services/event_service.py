"""
src/api/services/event_service.py

Server-Sent Events, derived from REAL IncidentEvent objects already
persisted by IncidentManager -- no synthetic/frontend-only events.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

EVENT_TYPE_MAP = {
    "DETECTION_CREATED": "DETECTION_COMPLETED",
    "QUANTUM_ROUTING_REQUESTED": "QUANTUM_ROUTING",
    "QUANTUM_VERIFICATION_COMPLETED": "QUANTUM_VERIFICATION",
    "QUANTUM_VERIFICATION_FAILED": "QUANTUM_VERIFICATION",
    "HYBRID_DECISION_CREATED": "HYBRID_DECISION",
    "RISK_ASSESSED": "RISK_ASSESSED",
    "DEFENSE_ACTION_EXECUTED": "DEFENSE_EXECUTED",
    "INCIDENT_RESOLVED": "INCIDENT_UPDATED",
    "INCIDENT_ESCALATED": "INCIDENT_UPDATED",
    "IDEMPOTENT_SKIP": "INCIDENT_UPDATED",
}


def format_sse(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def stream_incident_events(experiment_service, poll_interval: float = 1.0, max_iterations: int = 300) -> Iterator[str]:
    seen_event_ids = set()
    iterations = 0
    while iterations < max_iterations:
        events = experiment_service._event_store.read_all()
        for e in events:
            if e.event_id in seen_event_ids:
                continue
            seen_event_ids.add(e.event_id)
            frontend_type = EVENT_TYPE_MAP.get(e.event_type, e.event_type)
            yield format_sse({
                "event_type": frontend_type, "raw_event_type": e.event_type,
                "incident_id": e.incident_id, "correlation_id": e.correlation_id,
                "reason": e.reason, "timestamp": e.timestamp,
            }, event=frontend_type)
        iterations += 1
        time.sleep(poll_interval)
