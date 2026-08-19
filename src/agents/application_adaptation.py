"""
src/agents/application_adaptation.py

Deterministic adaptation policy: observation -> security outcome ->
feedback -> updated agent strategy -> next session. No LLM. Every
decision is a plain, inspectable rule over real outcome fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AdaptationRecord:
    previous_scenario_id: str
    escalated: bool
    risk_level: Optional[str]
    adaptation_decision: str
    next_scenario_id: str


class DeterministicAdaptationPolicy:
    """
    Rule (deterministic, inspectable):
      - If the previous session ESCALATED, the adversarial agent avoids
        repeating the SAME scenario next session (moves to the next
        scenario in its allow-list) -- a simple "back off after being
        caught" rule.
      - If the previous session did NOT escalate, the agent repeats the
        same scenario.
      - If only one scenario is allowed, it has no choice but to repeat
        it -- recorded explicitly, not a silent no-op.
    """

    def decide_next_scenario(
        self, allowed_scenario_ids: List[str], previous_scenario_id: str, escalated: bool, risk_level: Optional[str] = None,
    ) -> AdaptationRecord:
        if len(allowed_scenario_ids) <= 1:
            next_id = previous_scenario_id
            decision = "only one allowed scenario -- no alternative to switch to"
        elif escalated:
            idx = allowed_scenario_ids.index(previous_scenario_id)
            next_id = allowed_scenario_ids[(idx + 1) % len(allowed_scenario_ids)]
            decision = f"previous session escalated -- backing off from '{previous_scenario_id}' to '{next_id}'"
        else:
            next_id = previous_scenario_id
            decision = f"previous session did not escalate -- repeating '{previous_scenario_id}'"

        return AdaptationRecord(
            previous_scenario_id=previous_scenario_id, escalated=escalated, risk_level=risk_level,
            adaptation_decision=decision, next_scenario_id=next_id,
        )
