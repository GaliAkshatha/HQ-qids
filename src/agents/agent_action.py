"""
src/agents/agent_action.py

ApplicationAgentAction represents one application-interaction step (e.g.
"log in", "create an application record"), as opposed to
src/agents/agent_base.py's AgentAction, which represents "select an
NSL-KDD scenario to generate traffic from." These are deliberately
different classes with different names -- they serve different domains
(real application interaction vs. bounded-perturbation traffic
generation) and are never meant to be interchangeable.

action_type values are constrained to the real, verified Suzume actions
from the Stage A audit -- see ACTION_TYPES below. Nothing here invents
an endpoint or workflow that wasn't confirmed against the actual
apps/api/src/modules/*/*.routes.ts files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Every action type here corresponds to a route verified in Stage A's
# audit of https://github.com/GaliAkshatha/suzume. No invented endpoints.
ACTION_TYPES = {
    # auth
    "register", "login", "refresh", "logout", "get_me",
    "change_password", "forgot_password", "reset_password",
    # companies
    "list_companies", "create_company", "get_company",
    # applications
    "list_applications", "create_application", "get_application",
    "update_application", "delete_application",
    # rounds
    "list_rounds", "create_round", "get_round", "update_round", "delete_round",
    # experiences
    "list_experiences", "get_experience", "update_experience", "create_experience",
    # questions
    "create_question", "update_question", "delete_question",
    # learnings
    "list_learnings", "create_learning", "get_learning", "update_learning", "delete_learning",
    "create_action_item", "update_action_item", "delete_action_item",
    # preparation
    "list_preparation", "get_preparation_activity", "preparation_setup",
    "create_preparation_topic", "delete_preparation_topic",
    "list_preparation_logs", "create_preparation_log", "update_preparation_log", "delete_preparation_log",
    "list_preparation_sources", "create_preparation_source", "refresh_preparation_source", "delete_preparation_source",
    "update_preparation_topic_progress",
    # dashboard / calendar / analytics / extraction
    "get_dashboard_summary", "get_calendar_events", "get_analytics_overview", "extraction_parse",
}


@dataclass(frozen=True)
class ApplicationAgentAction:
    action_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    target_id: Optional[str] = None
    parent_id: Optional[str] = None

    def __post_init__(self):
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"Unknown action_type: '{self.action_type}'. Known: {sorted(ACTION_TYPES)}")
