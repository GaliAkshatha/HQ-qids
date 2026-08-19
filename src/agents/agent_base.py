"""
src/agents/agent_base.py

Agent is deterministic/policy-based, NOT LLM-powered, per the approved
Phase 8 scope -- act() is a pure function of (rng, context), with no
hidden state that makes behavior nondeterministic beyond the injected
random.Random. No agent implementation performs network, socket, or
subprocess operations; act() only returns a plain dataclass describing
which scenario to generate next.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.agents.contracts import AgentProfile


@dataclass(frozen=True)
class AgentAction:
    agent_id: str
    agent_type: str
    scenario_name: str
    session_id: str


class Agent(ABC):
    def __init__(self, profile: AgentProfile) -> None:
        self.profile = profile

    @abstractmethod
    def act(self, session_id: str, rng: random.Random) -> AgentAction:
        """Selects a scenario for this turn. Must only choose from
        self.profile.allowed_scenarios -- enforced by each concrete
        subclass, checked again by EnvironmentEngine as a second,
        independent layer."""
        raise NotImplementedError
