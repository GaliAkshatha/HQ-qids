"""
src/agents/feedback_listener.py

Strictly read-only. Uses XRANGE (a plain scan, not XREADGROUP) against
streams that already exist and are already published by the unmodified
incident-worker/defense-worker -- no new stream, no consumer group, no
ACK, and therefore no interference with the real StreamWorker consumers.

Per the approved constraint, this does NOT implement adaptive agent
behavior -- it only reads and returns outcomes for logging/metrics.
Nothing here writes back to Redis, an agent, or any domain component.

Only imports from src.runtime (to reach Redis) -- no detection/quantum/
routing/hybrid/defense/incident import.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis

from src.runtime.config import RuntimePolicyConfig


@dataclass
class TurnOutcome:
    correlation_id: str
    incident_current_state: Optional[str]
    escalated: Optional[bool]
    final_prediction: Optional[str]
    selected_action: Optional[str]
    risk_level: Optional[str]


class FeedbackListener:
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig) -> None:
        self.client = client
        self.config = config

    def _read_stream_matching(self, stream_key: str, field: str, value: str) -> List[Dict[str, Any]]:
        entries = self.client.xrange(stream_key, min="-", max="+")
        matches = []
        for _entry_id, fields in entries:
            data = json.loads(fields["data"])
            if data.get(field) == value:
                matches.append(data)
        return matches

    def read_incident_updates(self, incident_id: str) -> List[Dict[str, Any]]:
        return self._read_stream_matching(self.config.streams["incident_updated"], "incident_id", incident_id)

    def read_defense_completions(self, incident_id: str) -> List[Dict[str, Any]]:
        return self._read_stream_matching(self.config.streams["defense_completed"], "incident_id", incident_id)

    def collect_outcome(self, incident_id: str, correlation_id: Optional[str] = None) -> Optional[TurnOutcome]:
        """
        Merges the latest incident.updated and defense.completed entries
        for ONE incident_id (a single agent turn) into a read-only
        summary. incident_id, not correlation_id, is the correct
        per-turn key: when AgentSessionCorrelation is used, multiple
        turns share one correlation_id (that's the point -- it's what
        lets them correlate into one incident-repetition count), but
        incident_id is still assigned fresh per turn by the gateway
        regardless of correlation strategy. Filtering by correlation_id
        alone would silently conflate different turns' outcomes.
        """
        incident_updates = self.read_incident_updates(incident_id)
        defense_completions = self.read_defense_completions(incident_id)
        if not incident_updates and not defense_completions:
            return None

        latest_incident = incident_updates[-1] if incident_updates else {}
        latest_defense = defense_completions[-1] if defense_completions else {}
        defense_payload = latest_defense.get("payload", {})
        defense_result = defense_payload.get("defense_result", {})
        hybrid_decision = defense_payload.get("hybrid_decision", {})

        return TurnOutcome(
            correlation_id=correlation_id or latest_incident.get("correlation_id") or latest_defense.get("correlation_id"),
            incident_current_state=latest_incident.get("payload", {}).get("current_state"),
            escalated=latest_incident.get("payload", {}).get("escalated"),
            final_prediction=hybrid_decision.get("final_prediction"),
            selected_action=defense_result.get("action"),
            risk_level=defense_result.get("severity"),
        )

    def wait_for_outcome(self, incident_id: str, timeout_ms: int, poll_block_ms: int, correlation_id: Optional[str] = None) -> Optional[TurnOutcome]:
        """Polls (still read-only XRANGE scans, no blocking XREAD) up to
        timeout_ms for an outcome to appear."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            outcome = self.collect_outcome(incident_id, correlation_id)
            if outcome is not None and outcome.incident_current_state is not None:
                return outcome
            time.sleep(poll_block_ms / 1000.0)
        return self.collect_outcome(incident_id, correlation_id)

    def session_escalation_count(self, correlation_id: str) -> int:
        """Session-level view (unlike collect_outcome, this legitimately
        DOES aggregate by correlation_id/session): counts how many
        distinct incidents within this session reached an escalated
        state -- used by the repeated-incident experiment."""
        entries = self._read_stream_matching(self.config.streams["incident_updated"], "correlation_id", correlation_id)
        escalated_incident_ids = {
            e["incident_id"] for e in entries if e.get("payload", {}).get("escalated") is True
        }
        return len(escalated_incident_ids)
