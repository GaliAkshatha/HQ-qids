"""
src/agents/agent_session.py

Holds per-session state including the access token, IN MEMORY ONLY.
Nothing in this class is ever serialized into an ApplicationObservation
or logged -- access_token exists here purely so
SuzumeTrafficSource.execute_action() can attach it as an Authorization
header for subsequent calls within the same session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentSession:
    session_id: str
    agent_id: str
    agent_type: str
    target_label: str
    created_at: str

    access_token: Optional[str] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    # Redacted response body from the most recent call -- populated by
    # ApplicationTrafficSource implementations with accessToken (and any
    # other token/secret field) stripped, so a workflow-driven agent can
    # read e.g. the id of a just-created resource to chain the next
    # action, without any secret ever passing through agent-visible state.
    last_response_data: Optional[dict] = None

    _sequence_counter: int = field(default=0, repr=False)

    def next_sequence(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def is_authenticated(self) -> bool:
        return self.access_token is not None

    def set_authenticated(self, access_token: str, user_id: str, user_email: str) -> None:
        self.access_token = access_token
        self.user_id = user_id
        self.user_email = user_email

    def clear_authentication(self) -> None:
        self.access_token = None
        self.user_id = None
        self.user_email = None
