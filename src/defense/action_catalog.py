"""
src/defense/action_catalog.py

Defines the fixed vocabulary of simulated defense actions and their
static metadata (reversibility, rollback action). This is data, not
behavior -- the simulator/executor decide what actually happens to state;
this module just says what's structurally true about each action type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


MONITOR = "MONITOR"
INCREASE_MONITORING = "INCREASE_MONITORING"
RATE_LIMIT = "RATE_LIMIT"
ISOLATE_SIMULATED_SOURCE = "ISOLATE_SIMULATED_SOURCE"
BLOCK_SIMULATED_SOURCE = "BLOCK_SIMULATED_SOURCE"
TERMINATE_SIMULATED_SESSION = "TERMINATE_SIMULATED_SESSION"

ALL_ACTIONS = {
    MONITOR,
    INCREASE_MONITORING,
    RATE_LIMIT,
    ISOLATE_SIMULATED_SOURCE,
    BLOCK_SIMULATED_SOURCE,
    TERMINATE_SIMULATED_SESSION,
}


@dataclass(frozen=True)
class ActionSpec:
    action_type: str
    reversible: bool
    rollback_action: Optional[str]  # None if not reversible
    expected_effect: str  # human-readable description of the intended state transition


_CATALOG: Dict[str, ActionSpec] = {
    MONITOR: ActionSpec(
        action_type=MONITOR, reversible=True, rollback_action=None,
        expected_effect="target flagged for baseline monitoring; no state restriction applied",
    ),
    INCREASE_MONITORING: ActionSpec(
        action_type=INCREASE_MONITORING, reversible=True, rollback_action="DECREASE_MONITORING",
        expected_effect="target's monitoring_level increased",
    ),
    RATE_LIMIT: ActionSpec(
        action_type=RATE_LIMIT, reversible=True, rollback_action="REMOVE_RATE_LIMIT",
        expected_effect="target's rate_limited flag set to True",
    ),
    ISOLATE_SIMULATED_SOURCE: ActionSpec(
        action_type=ISOLATE_SIMULATED_SOURCE, reversible=True, rollback_action="RESTORE_SOURCE_ACCESS",
        expected_effect="target's isolated flag set to True",
    ),
    BLOCK_SIMULATED_SOURCE: ActionSpec(
        action_type=BLOCK_SIMULATED_SOURCE, reversible=True, rollback_action="UNBLOCK_SIMULATED_SOURCE",
        expected_effect="target's blocked flag set to True",
    ),
    TERMINATE_SIMULATED_SESSION: ActionSpec(
        action_type=TERMINATE_SIMULATED_SESSION, reversible=False, rollback_action=None,
        expected_effect="target's active_sessions cleared; sessions cannot be un-terminated, only re-established",
    ),
}


def get_action_spec(action_type: str) -> ActionSpec:
    if action_type not in _CATALOG:
        raise ValueError(f"Unknown action type: '{action_type}'. Known actions: {sorted(ALL_ACTIONS)}")
    return _CATALOG[action_type]


def is_known_action(action_type: str) -> bool:
    return action_type in _CATALOG
