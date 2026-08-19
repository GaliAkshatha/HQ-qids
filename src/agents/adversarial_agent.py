"""
src/agents/adversarial_agent.py

The safety boundary lives here, enforced in code, not merely documented:
AdversarialAgent can only ever be constructed with a set of allowed
scenarios, and act() can only select from that set. There is no method
on this class, or anywhere in src/agents/, that performs a real network,
socket, or subprocess operation -- "adversarial" here means "selects a
scenario whose category is 'attack'," nothing more.
"""

from __future__ import annotations

import random

from src.agents.agent_base import Agent, AgentAction
from src.agents.contracts import AgentProfile


class ScenarioNotAllowedError(ValueError):
    pass


class AdversarialAgent(Agent):
    def __init__(self, agent_id: str, allowed_scenarios: tuple) -> None:
        if not allowed_scenarios:
            raise ScenarioNotAllowedError("AdversarialAgent requires a non-empty allowed_scenarios allow-list")
        super().__init__(AgentProfile(agent_id=agent_id, agent_type="adversarial", allowed_scenarios=allowed_scenarios))

    def act(self, session_id: str, rng: random.Random) -> AgentAction:
        scenario_name = rng.choice(self.profile.allowed_scenarios)
        if scenario_name not in self.profile.allowed_scenarios:
            raise ScenarioNotAllowedError(
                f"Scenario '{scenario_name}' is not in this agent's allow-list: {self.profile.allowed_scenarios}"
            )
        return AgentAction(
            agent_id=self.profile.agent_id, agent_type=self.profile.agent_type,
            scenario_name=scenario_name, session_id=session_id,
        )

    def request_scenario(self, scenario_name: str, session_id: str) -> AgentAction:
        """Explicit, validated request path (as opposed to act()'s random
        selection) -- used by tests and by EnvironmentEngine when a
        specific scenario is needed. Still enforces the allow-list."""
        if scenario_name not in self.profile.allowed_scenarios:
            raise ScenarioNotAllowedError(
                f"Scenario '{scenario_name}' is not in this agent's allow-list: {self.profile.allowed_scenarios}"
            )
        return AgentAction(
            agent_id=self.profile.agent_id, agent_type=self.profile.agent_type,
            scenario_name=scenario_name, session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Stage B addition: SuzumeAdversarialAgent -- bounded, safe, application-
# level adversarial behavior against Suzume, added additively alongside
# AdversarialAgent above (which is untouched). Every scenario is a real,
# fully-documented AdversarialScenario with a strict max_actions bound.
# No credential stuffing, no destructive action, no flooding, no real
# denial-of-service, nothing against any third party -- every scenario
# below operates only on an account the agent itself just registered.
# ---------------------------------------------------------------------------

import uuid as _uuid
from dataclasses import dataclass as _dataclass
from typing import Dict as _Dict, List as _List, Optional as _Optional

from src.agents.agent_action import ApplicationAgentAction
from src.agents.agent_session import AgentSession
from src.agents.traffic_source import ApplicationTrafficSource


@_dataclass(frozen=True)
class AdversarialScenario:
    scenario_id: str
    description: str
    max_actions: int
    reason: str
    expected_telemetry_pattern: str


SUZUME_ADVERSARIAL_SCENARIOS: _Dict[str, AdversarialScenario] = {
    "repeated_failed_login": AdversarialScenario(
        scenario_id="repeated_failed_login",
        description="Attempt login with an incorrect password a bounded number of times, against an account the agent itself just registered -- never a real third party's account.",
        max_actions=5,
        reason="Exercises failed_auth_rate/auth_failure_burst detection without real credential-stuffing at scale.",
        expected_telemetry_pattern="Consecutive 401s on POST /api/auth/login; failed_auth_rate and auth_failure_burst both near 1.0; low session_action_entropy.",
    ),
    "malformed_payload_probe": AdversarialScenario(
        scenario_id="malformed_payload_probe",
        description="Submit schema-invalid application-creation payloads (missing/empty required fields, invalid enum values) -- all rejected by validation before touching any data.",
        max_actions=5,
        reason="Exercises validation_failure_rate detection safely -- every payload is expected to be rejected by Suzume's own zod validation, never persisted.",
        expected_telemetry_pattern="Repeated 400 responses with error.code=VALIDATION_ERROR; validation_failure_rate near 1.0.",
    ),
    "invalid_resource_probe": AdversarialScenario(
        scenario_id="invalid_resource_probe",
        description="Request applications/rounds/experiences by randomly-generated, essentially-certain-to-not-exist UUIDs -- pure reads, no mutation attempted.",
        max_actions=5,
        reason="Exercises invalid_resource_rate detection -- read-only, non-destructive, bounded ID-guessing pattern.",
        expected_telemetry_pattern="Repeated 404 NOT_FOUND responses; invalid_resource_rate near 1.0.",
    ),
    "rapid_endpoint_switching": AdversarialScenario(
        scenario_id="rapid_endpoint_switching",
        description="Rapidly call a bounded sequence of different READ-ONLY endpoints in quick succession -- no create/update/delete calls in this scenario at all.",
        max_actions=6,
        reason="Exercises endpoint_switch_rate/request_rate/session_action_entropy detection with zero risk of data modification.",
        expected_telemetry_pattern="endpoint_switch_rate near 1.0, elevated request_rate, high session_action_entropy (many distinct action types).",
    ),
}

_READ_ONLY_ENDPOINT_CYCLE = [
    "list_applications", "get_dashboard_summary", "list_learnings",
    "get_analytics_overview", "get_calendar_events", "list_companies",
]


class SuzumeAdversarialAgent:
    def __init__(self, agent_id: str, allowed_scenario_ids: tuple, seed: _Optional[int] = None) -> None:
        if not allowed_scenario_ids:
            raise ScenarioNotAllowedError("SuzumeAdversarialAgent requires a non-empty allowed_scenario_ids allow-list")
        for sid in allowed_scenario_ids:
            if sid not in SUZUME_ADVERSARIAL_SCENARIOS:
                raise ScenarioNotAllowedError(f"Unknown scenario_id: '{sid}'. Known: {sorted(SUZUME_ADVERSARIAL_SCENARIOS)}")
        self.agent_id = agent_id
        self.agent_type = "adversarial"
        self.allowed_scenario_ids = allowed_scenario_ids
        self.rng = random.Random(seed)

    def run_session(
        self, traffic_source: ApplicationTrafficSource, session: AgentSession, scenario_id: _Optional[str] = None
    ) -> _List:
        if scenario_id is None:
            scenario_id = self.rng.choice(self.allowed_scenario_ids)
        if scenario_id not in self.allowed_scenario_ids:
            raise ScenarioNotAllowedError(
                f"Scenario '{scenario_id}' is not in this agent's allow-list: {self.allowed_scenario_ids}"
            )
        scenario = SUZUME_ADVERSARIAL_SCENARIOS[scenario_id]

        observations = []

        def do(action: ApplicationAgentAction):
            obs = traffic_source.execute_action(session, action)
            observations.append(obs)
            return obs

        unique_suffix = _uuid.uuid4().hex[:10]
        email = f"adversarial-agent-{unique_suffix}@example.com"
        real_password = "AgentPassword123"
        do(ApplicationAgentAction(action_type="register", payload={"name": "Adversarial Agent", "email": email, "password": real_password}))

        if scenario_id == "repeated_failed_login":
            for _ in range(scenario.max_actions):
                do(ApplicationAgentAction(action_type="login", payload={"email": email, "password": "DefinitelyWrongPassword999"}))

        elif scenario_id == "malformed_payload_probe":
            bad_payloads = [
                {},  # missing everything
                {"companyName": ""},  # empty required field
                {"role": ""},  # missing companyName, empty role
                {"companyName": "X", "role": "Y", "status": "NOT_A_REAL_STATUS"},  # invalid enum
                {"companyName": "X", "role": "Y", "internship": "not-a-boolean"},  # wrong type
            ]
            for payload in bad_payloads[: scenario.max_actions]:
                do(ApplicationAgentAction(action_type="create_application", payload=payload))

        elif scenario_id == "invalid_resource_probe":
            for _ in range(scenario.max_actions):
                fake_id = str(_uuid.uuid4())
                do(ApplicationAgentAction(action_type="get_application", target_id=fake_id))

        elif scenario_id == "rapid_endpoint_switching":
            for action_type in _READ_ONLY_ENDPOINT_CYCLE[: scenario.max_actions]:
                do(ApplicationAgentAction(action_type=action_type))

        return observations
