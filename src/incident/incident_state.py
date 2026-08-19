"""
src/incident/incident_state.py

Thin validation layer around the transition table defined in
src/contracts/incident.py (the single source of truth for what states
and edges exist). No transition happens without explicit
previous_state/new_state/reason/triggering_event -- an invalid edge
raises rather than being silently allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts.incident import INCIDENT_STATES, VALID_TRANSITIONS


class InvalidTransitionError(ValueError):
    pass


@dataclass
class TransitionRequest:
    incident_id: str
    previous_state: str
    new_state: str
    reason: str
    triggering_event: str  # event_type from src/contracts/incident.py's EVENT_TYPES


def validate_transition(request: TransitionRequest) -> None:
    """Raises InvalidTransitionError if the edge is not permitted. Never
    silently allows an unlisted transition."""
    if request.previous_state not in INCIDENT_STATES:
        raise InvalidTransitionError(f"Unknown previous_state: '{request.previous_state}'")
    if request.new_state not in INCIDENT_STATES:
        raise InvalidTransitionError(f"Unknown new_state: '{request.new_state}'")

    allowed = VALID_TRANSITIONS.get(request.previous_state, set())
    if request.new_state not in allowed:
        raise InvalidTransitionError(
            f"Invalid transition for incident '{request.incident_id}': "
            f"'{request.previous_state}' -> '{request.new_state}' is not permitted. "
            f"Allowed from '{request.previous_state}': {sorted(allowed) or '(terminal, no transitions allowed)'}"
        )
    if not request.reason:
        raise InvalidTransitionError("reason is required for every transition -- no silent transitions")
    if not request.triggering_event:
        raise InvalidTransitionError("triggering_event is required for every transition -- no silent transitions")


def reconstruct_snapshot(incident_id: str, events: list) -> "IncidentSnapshot | None":
    """
    Folds an incident's ordered event history into its current
    IncidentSnapshot -- this is the literal state-reconstruction-from-
    events operation. Returns None if no events exist for this
    incident_id (nothing to reconstruct).

    This function is what makes the JSONL file genuinely the source of
    truth: a brand-new process with no in-memory state can call this
    against events read from disk and get back the exact same snapshot
    the original process would have held in memory.
    """
    from src.contracts.incident import ESCALATED, IncidentSnapshot, INCIDENT_ESCALATED

    incident_events = [e for e in events if e.incident_id == incident_id]
    if not incident_events:
        return None

    first = incident_events[0]
    last = incident_events[-1]

    current_state = first.previous_state or first.new_state
    escalation_reasons: list = []
    for e in incident_events:
        if e.new_state is not None:
            current_state = e.new_state
        if e.event_type == INCIDENT_ESCALATED:
            reasons_from_payload = e.payload.get("escalation_reasons")
            if reasons_from_payload:
                escalation_reasons.extend(reasons_from_payload)
            else:
                escalation_reasons.append(e.reason)

    return IncidentSnapshot(
        incident_id=incident_id,
        correlation_id=first.correlation_id,
        current_state=current_state,
        created_at=first.timestamp,
        updated_at=last.timestamp,
        event_ids=[e.event_id for e in incident_events],
        escalated=current_state == ESCALATED,
        escalation_reasons=escalation_reasons,
    )
