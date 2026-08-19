"""
src/agents/suzume_traffic_source.py

Real HTTP client against a Suzume deployment (or local controlled
target), implementing exactly the routes verified in the Stage A audit
of https://github.com/GaliAkshatha/suzume -- no invented endpoints.

Auth flow mirrors apps/web/src/services/api/client.ts exactly: bearer
access token in memory (held on the AgentSession, never in telemetry),
httpOnly refresh cookie handled transparently by requests.Session's
cookie jar (never read or logged), one auto-refresh-and-retry on a 401
for non-auth endpoints -- same as the real frontend.

target_label is a required constructor argument (REAL_SUZUME_INTERACTION
or CONTROLLED_LOCAL_SUZUME) so results are never silently mislabeled.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import requests

from src.agents.agent_action import ApplicationAgentAction
from src.agents.agent_session import AgentSession
from src.agents.application_observation import ApplicationObservation
from src.agents.traffic_source import ApplicationTrafficSource

ROUTE_MAP: Dict[str, Tuple[str, str, bool]] = {
    "register": ("POST", "/auth/register", False),
    "login": ("POST", "/auth/login", False),
    "refresh": ("POST", "/auth/refresh", False),
    "logout": ("POST", "/auth/logout", False),
    "get_me": ("GET", "/auth/me", True),
    "change_password": ("POST", "/auth/change-password", True),
    "forgot_password": ("POST", "/auth/forgot-password", False),
    "reset_password": ("POST", "/auth/reset-password", False),

    "list_companies": ("GET", "/companies", True),
    "create_company": ("POST", "/companies", True),
    "get_company": ("GET", "/companies/{id}", True),

    "list_applications": ("GET", "/applications", True),
    "create_application": ("POST", "/applications", True),
    "get_application": ("GET", "/applications/{id}", True),
    "update_application": ("PATCH", "/applications/{id}", True),
    "delete_application": ("DELETE", "/applications/{id}", True),

    "list_rounds": ("GET", "/applications/{parent_id}/rounds", True),
    "create_round": ("POST", "/applications/{parent_id}/rounds", True),
    "get_round": ("GET", "/rounds/{id}", True),
    "update_round": ("PATCH", "/rounds/{id}", True),
    "delete_round": ("DELETE", "/rounds/{id}", True),

    "list_experiences": ("GET", "/experiences", True),
    "get_experience": ("GET", "/experiences/{id}", True),
    "update_experience": ("PATCH", "/experiences/{id}", True),
    "create_experience": ("POST", "/rounds/{parent_id}/experience", True),

    "create_question": ("POST", "/experiences/{parent_id}/questions", True),
    "update_question": ("PATCH", "/questions/{id}", True),
    "delete_question": ("DELETE", "/questions/{id}", True),

    "list_learnings": ("GET", "/learnings", True),
    "create_learning": ("POST", "/learnings", True),
    "get_learning": ("GET", "/learnings/{id}", True),
    "update_learning": ("PATCH", "/learnings/{id}", True),
    "delete_learning": ("DELETE", "/learnings/{id}", True),
    "create_action_item": ("POST", "/learnings/{parent_id}/actions", True),
    "update_action_item": ("PATCH", "/actions/{id}", True),
    "delete_action_item": ("DELETE", "/actions/{id}", True),

    "list_preparation": ("GET", "/preparation", True),
    "get_preparation_activity": ("GET", "/preparation/activity", True),
    "preparation_setup": ("POST", "/preparation/setup", True),
    "create_preparation_topic": ("POST", "/preparation/topics", True),
    "delete_preparation_topic": ("DELETE", "/preparation/topics/{id}", True),
    "list_preparation_logs": ("GET", "/preparation/logs", True),
    "create_preparation_log": ("POST", "/preparation/logs", True),
    "update_preparation_log": ("PATCH", "/preparation/logs/{id}", True),
    "delete_preparation_log": ("DELETE", "/preparation/logs/{id}", True),
    "list_preparation_sources": ("GET", "/preparation/sources", True),
    "create_preparation_source": ("POST", "/preparation/sources", True),
    "refresh_preparation_source": ("POST", "/preparation/sources/{id}/refresh", True),
    "delete_preparation_source": ("DELETE", "/preparation/sources/{id}", True),
    "update_preparation_topic_progress": ("PATCH", "/preparation/{id}", True),

    "get_dashboard_summary": ("GET", "/dashboard/summary", True),
    "get_calendar_events": ("GET", "/calendar/events", True),
    "get_analytics_overview": ("GET", "/analytics/overview", True),
    "extraction_parse": ("POST", "/extraction/parse", True),
}

_AUTH_ISSUING_ACTIONS = {"register", "login", "refresh"}
_REDACTED_RESPONSE_KEYS = {"accessToken", "refreshToken", "passwordHash", "password"}


def _redact(data):
    if not isinstance(data, dict):
        return data
    return {k: v for k, v in data.items() if k not in _REDACTED_RESPONSE_KEYS}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SuzumeTrafficSource(ApplicationTrafficSource):
    def __init__(self, base_url: str, target_label: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.target_label = target_label
        self.timeout_seconds = timeout_seconds
        self._http_sessions: Dict[str, requests.Session] = {}

    def _http_session_for(self, session: AgentSession) -> requests.Session:
        if session.session_id not in self._http_sessions:
            self._http_sessions[session.session_id] = requests.Session()
        return self._http_sessions[session.session_id]

    def _resolve_path(self, action: ApplicationAgentAction) -> str:
        _, template, _ = ROUTE_MAP[action.action_type]
        path = template
        if "{id}" in path:
            if not action.target_id:
                raise ValueError(f"action_type '{action.action_type}' requires target_id")
            path = path.replace("{id}", action.target_id)
        if "{parent_id}" in path:
            if not action.parent_id:
                raise ValueError(f"action_type '{action.action_type}' requires parent_id")
            path = path.replace("{parent_id}", action.parent_id)
        return f"/api{path}"

    def execute_action(self, session: AgentSession, action: ApplicationAgentAction) -> ApplicationObservation:
        if action.action_type not in ROUTE_MAP:
            raise ValueError(f"No route mapping for action_type: '{action.action_type}'")

        method, _, _ = ROUTE_MAP[action.action_type]
        endpoint = self._resolve_path(action)
        return self._perform_request(session, action, method, endpoint, allow_refresh_retry=True)

    def _perform_request(
        self, session: AgentSession, action: ApplicationAgentAction, method: str, endpoint: str, allow_refresh_retry: bool
    ) -> ApplicationObservation:
        url = f"{self.base_url}{endpoint}"
        http_session = self._http_session_for(session)

        headers = {"Content-Type": "application/json"}
        if session.access_token:
            headers["Authorization"] = f"Bearer {session.access_token}"

        sequence_number = session.next_sequence()
        request_kwargs: Dict[str, object] = {"headers": headers, "timeout": self.timeout_seconds}
        if method == "GET":
            if action.payload:
                request_kwargs["params"] = action.payload
        else:
            request_kwargs["json"] = action.payload

        error: Optional[str] = None
        status_code: Optional[int] = None
        response_size: Optional[int] = None
        validation_success = True
        validation_errors = None
        response_json = None

        t0 = time.perf_counter()
        try:
            response = http_session.request(method, url, **request_kwargs)
            status_code = response.status_code
            response_size = len(response.content) if response.content else 0
            try:
                response_json = response.json()
            except ValueError:
                response_json = None

            if status_code == 400 and isinstance(response_json, dict):
                err = response_json.get("error") or {}
                if err.get("code") == "VALIDATION_ERROR":
                    validation_success = False
                    validation_errors = err.get("details")

            if isinstance(response_json, dict):
                session.last_response_data = _redact(response_json)

            if action.action_type in _AUTH_ISSUING_ACTIONS and status_code is not None and status_code < 300 and response_json:
                access_token = response_json.get("accessToken")
                user = response_json.get("user") or {}
                if access_token:
                    session.set_authenticated(access_token, user.get("id", ""), user.get("email", ""))
            if action.action_type == "logout" and status_code is not None and status_code < 300:
                session.clear_authentication()

            if status_code == 401 and allow_refresh_retry and not endpoint.startswith("/api/auth/"):
                refresh_action = ApplicationAgentAction(action_type="refresh")
                refresh_obs = self._perform_request(session, refresh_action, "POST", "/api/auth/refresh", allow_refresh_retry=False)
                if refresh_obs.status_code is not None and refresh_obs.status_code < 300:
                    return self._perform_request(session, action, method, endpoint, allow_refresh_retry=False)

        except requests.exceptions.RequestException as e:
            error = type(e).__name__

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ApplicationObservation(
            timestamp=_now_iso(), session_id=session.session_id, agent_id=session.agent_id,
            agent_type=session.agent_type, action_type=action.action_type, method=method,
            endpoint=endpoint, status_code=status_code, latency_ms=round(latency_ms, 3),
            authenticated=session.is_authenticated(), validation_success=validation_success,
            validation_errors=validation_errors, response_size=response_size,
            target_label=self.target_label, sequence_number=sequence_number, error=error,
        )
