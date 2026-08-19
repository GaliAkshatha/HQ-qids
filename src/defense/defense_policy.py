"""
src/defense/defense_policy.py

Loads config/defense_policy.json and selects an action for a given
(risk_level, decision_status) pair. HybridDecision.decision_status is the
approved tiebreaker between the milder and stronger action listed for
HIGH/CRITICAL in the spec -- "confirmed" (solid evidence, per Phase 4's
own definition) gets the stronger option; "uncertain" or "normal" gets
the milder, reversible one. This does not choose the strongest action
merely because risk is elevated -- it's keyed off evidence quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "defense_policy.json"

VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_DECISION_STATUSES = {"confirmed", "uncertain", "normal"}


@dataclass
class DefensePolicyConfig:
    risk_action_map: Dict[str, Dict[str, str]]
    disabled_actions: List[str]
    uncertain_requires_reversible_action: bool
    max_recovery_retries: int
    rollback_on_failure: bool
    simulation_mode: bool

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "DefensePolicyConfig":
        with open(path) as f:
            raw = json.load(f)
        cls._validate_raw(raw)
        recovery = raw["recovery"]
        return cls(
            risk_action_map=raw["risk_action_map"],
            disabled_actions=raw.get("disabled_actions", []),
            uncertain_requires_reversible_action=raw["uncertain_requires_reversible_action"],
            max_recovery_retries=recovery["max_retries"],
            rollback_on_failure=recovery["rollback_on_failure"],
            simulation_mode=raw["simulation_mode"],
        )

    @staticmethod
    def _validate_raw(raw: Dict[str, Any]) -> None:
        risk_map = raw.get("risk_action_map")
        if not risk_map:
            raise ValueError("defense_policy.json missing 'risk_action_map'")
        missing_levels = VALID_RISK_LEVELS - set(risk_map.keys())
        if missing_levels:
            raise ValueError(f"risk_action_map is missing entries for: {missing_levels}")
        for level, by_status in risk_map.items():
            missing_statuses = VALID_DECISION_STATUSES - set(by_status.keys())
            if missing_statuses:
                raise ValueError(f"risk_action_map['{level}'] is missing decision_status entries for: {missing_statuses}")

    def select_action(self, risk_level: str, decision_status: str) -> str:
        """
        Deterministic action selection. Raises on unknown risk_level or
        decision_status rather than silently defaulting -- an unmapped
        combination is a configuration bug, not something to guess at.
        """
        if risk_level not in self.risk_action_map:
            raise ValueError(f"No policy entry for risk_level='{risk_level}'")
        by_status = self.risk_action_map[risk_level]
        if decision_status not in by_status:
            raise ValueError(f"No policy entry for risk_level='{risk_level}', decision_status='{decision_status}'")
        return by_status[decision_status]

    def policy_reason(self, risk_level: str, decision_status: str, action: str) -> str:
        return f"risk_level={risk_level}, decision_status={decision_status} -> {action} (config/defense_policy.json)"
