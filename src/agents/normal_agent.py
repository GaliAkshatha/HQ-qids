"""
src/agents/normal_agent.py

Deterministic, policy-based -- selects uniformly at random (via the
injected rng, not any hidden source of randomness) among its allowed
(benign) scenarios.
"""

from __future__ import annotations

import random

from src.agents.agent_base import Agent, AgentAction
from src.agents.contracts import AgentProfile


class NormalUserAgent(Agent):
    def __init__(self, agent_id: str, allowed_scenarios: tuple) -> None:
        super().__init__(AgentProfile(agent_id=agent_id, agent_type="normal", allowed_scenarios=allowed_scenarios))

    def act(self, session_id: str, rng: random.Random) -> AgentAction:
        scenario_name = rng.choice(self.profile.allowed_scenarios)
        return AgentAction(
            agent_id=self.profile.agent_id, agent_type=self.profile.agent_type,
            scenario_name=scenario_name, session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Stage B addition: SuzumeNormalAgent -- a REAL application-workflow agent,
# added additively alongside NormalUserAgent above (which is untouched).
# Does not subclass Agent/AgentAction (those are the NSL-KDD-scenario
# shapes) -- uses ApplicationAgentAction/AgentSession/ApplicationObservation
# instead, since this is a different domain.
# ---------------------------------------------------------------------------

import uuid as _uuid
from typing import List, Optional

from src.agents.agent_action import ApplicationAgentAction
from src.agents.agent_session import AgentSession
from src.agents.traffic_source import ApplicationTrafficSource


class SuzumeNormalAgent:
    """
    Deterministic, policy-based agent following the REAL, Stage-A-verified
    Suzume workflow:

        register -> dashboard summary -> list applications
        -> create an application -> create a round for it
        -> record an experience for that round -> add a question
        -> add a learning -> view analytics overview

    Every action_type here corresponds to a route confirmed in Stage A's
    audit (see src/agents/agent_action.py's ACTION_TYPES). Reproducible
    via a seed: the seed drives which field VALUES are generated (company
    name, role, from small fixed pools) -- the SEQUENCE of actions itself
    is fixed, matching the one real verified workflow, not randomized.
    """

    def __init__(self, agent_id: str, seed: Optional[int] = None) -> None:
        self.agent_id = agent_id
        self.agent_type = "normal"
        self.rng = random.Random(seed)

    def run_session(self, traffic_source: ApplicationTrafficSource, session: AgentSession) -> List:
        observations = []

        def do(action: ApplicationAgentAction):
            obs = traffic_source.execute_action(session, action)
            observations.append(obs)
            return obs

        unique_suffix = _uuid.uuid4().hex[:10]
        email = f"normal-agent-{unique_suffix}@example.com"
        company = self.rng.choice(["Acme Corp", "Initech", "Globex", "Umbrella Labs", "Stark Industries"])
        role = self.rng.choice(["SDE Intern", "Backend Developer", "SWE Intern", "Data Analyst"])
        round_title = self.rng.choice(["Online Assessment", "Technical Interview", "HR Round"])
        round_type = self.rng.choice(["ONLINE_ASSESSMENT", "TECHNICAL_INTERVIEW", "HR_ROUND"])

        do(ApplicationAgentAction(action_type="register", payload={
            "name": "Normal Agent", "email": email, "password": "AgentPassword123",
        }))
        do(ApplicationAgentAction(action_type="get_dashboard_summary"))
        do(ApplicationAgentAction(action_type="list_applications"))

        app_obs = do(ApplicationAgentAction(action_type="create_application", payload={
            "companyName": company, "role": role, "status": "APPLIED",
        }))
        application_id = None
        if session.last_response_data and "application" in session.last_response_data:
            application_id = session.last_response_data["application"].get("id")

        if application_id:
            round_obs = do(ApplicationAgentAction(
                action_type="create_round", parent_id=application_id,
                payload={"title": round_title, "type": round_type, "status": "COMPLETED"},
            ))
            round_id = None
            if session.last_response_data and "round" in session.last_response_data:
                round_id = session.last_response_data["round"].get("id")

            if round_id:
                exp_obs = do(ApplicationAgentAction(
                    action_type="create_experience", parent_id=round_id,
                    payload={"summary": "Went reasonably well.", "confidence": self.rng.randint(5, 9)},
                ))
                experience_id = None
                if session.last_response_data and "experience" in session.last_response_data:
                    experience_id = session.last_response_data["experience"].get("id")

                if experience_id:
                    do(ApplicationAgentAction(
                        action_type="create_question", parent_id=experience_id,
                        payload={"question": "Explain how a hash map works.", "category": "DSA"},
                    ))

        do(ApplicationAgentAction(action_type="create_learning", payload={
            "title": "Review system design basics", "category": "TECHNICAL",
        }))
        do(ApplicationAgentAction(action_type="get_analytics_overview"))

        return observations
