"""
src/agents/traffic_source.py

ApplicationTrafficSource is the seam between an agent and a real (or
locally controlled, or synthetic) application. Concrete implementations
(SuzumeTrafficSource, a future local test-target adapter) translate an
ApplicationAgentAction into a real HTTP call and return an
ApplicationObservation -- never a security-domain object. This module
imports nothing from src.detection/src.quantum/src.routing/src.hybrid/
src.defense/src.incident, matching the existing Phase 8 import boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.agents.agent_action import ApplicationAgentAction
from src.agents.agent_session import AgentSession
from src.agents.application_observation import ApplicationObservation


class ApplicationTrafficSource(ABC):
    @abstractmethod
    def execute_action(self, session: AgentSession, action: ApplicationAgentAction) -> ApplicationObservation:
        """Performs the real interaction (HTTP call or equivalent) for
        this action and returns the resulting telemetry. Must update
        session.access_token in place on successful login/register/
        refresh -- never returns a token inside the ApplicationObservation."""
        raise NotImplementedError
