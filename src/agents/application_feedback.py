"""
src/agents/application_feedback.py

Connects the existing (Phase 8, unmodified) FeedbackListener to the new
application-agent abstraction, strictly read-only, per Stage B point 9.

IMPORTANT SCOPE NOTE: per the model-boundary decision
(docs/APPLICATION_SECURITY_MODEL_BOUNDARY.md), application telemetry does
NOT currently flow into the existing incident pipeline -- there is no
Application Detection Model yet to produce a RiskAssessment/DefenseResult/
IncidentSnapshot from an ApplicationFeatureVector. This module therefore
does not (and cannot yet, honestly) wire a real Suzume interaction to a
real incident outcome end-to-end.

What this module DOES provide, correctly and testably: the exact
connection mechanism a future Application Detection Model's incident
pipeline would use, reusing FeedbackListener completely unmodified. It is
tested against real Redis with synthetic incident_ids (the same pattern
Phase 8's own FeedbackListener tests already use) to prove the wiring
itself works -- not to claim a real Suzume-driven incident exists yet.
"""

from __future__ import annotations

from typing import Optional

from src.agents.feedback_listener import FeedbackListener, TurnOutcome


def get_feedback_for_incident(incident_id: str, listener: FeedbackListener, correlation_id: Optional[str] = None) -> Optional[TurnOutcome]:
    """Thin, read-only pass-through to FeedbackListener.collect_outcome().
    Exists as the named connection point between the application-agent
    layer and the existing feedback mechanism -- adds no new behavior,
    modifies nothing, and performs no write of any kind."""
    return listener.collect_outcome(incident_id, correlation_id=correlation_id)
