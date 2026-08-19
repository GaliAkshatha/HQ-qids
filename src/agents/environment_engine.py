"""
src/agents/environment_engine.py

Orchestrates a run of N turns across a population of agents. Its ONLY
touchpoint with the existing system is TrafficGateway.ingest(sample_id,
raw_sample) -- the exact, unmodified integration seam. It does not
import detection/quantum/routing/hybrid/defense/incident modules.

sample_id is constructed as "<session_id>::<turn_index>::<short_uuid>"
per session_correlation.py's documented convention, so
AgentSessionCorrelation can parse the session key back out without any
change to TrafficGateway.ingest()'s signature.

Every generated sample's full provenance (source exemplar, scenario,
agent, session, perturbation, intended label) is logged via the existing
observability infrastructure before ingestion -- satisfying the
auditability requirement independent of anything downstream.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from src.agents.adversarial_agent import AdversarialAgent
from src.agents.agent_base import Agent, AgentAction
from src.agents.contracts import GeneratedTrafficRecord
from src.agents.normal_agent import NormalUserAgent
from src.agents.scenario_catalog import EnvironmentPolicy, ScenarioCatalog
from src.agents.session_correlation import SESSION_DELIMITER
from src.agents.templates import ExemplarBank, generate_sample
from src.observability.logging_config import get_logger, log_event

logger = get_logger("agent_environment")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnResult:
    sample_id: str
    session_id: str
    raw_sample: dict
    record: GeneratedTrafficRecord
    pipeline_message: object


class EnvironmentEngine:
    def __init__(
        self,
        gateway,
        catalog: ScenarioCatalog,
        policy: EnvironmentPolicy,
        exemplar_bank: Optional[ExemplarBank] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.gateway = gateway
        self.catalog = catalog
        self.policy = policy
        self.exemplar_bank = exemplar_bank or ExemplarBank()
        self.rng = random.Random(seed)
        self.records: List[GeneratedTrafficRecord] = []

    def build_default_agents(self) -> List[Agent]:
        normal = NormalUserAgent("normal-agent-1", self.policy.normal_allowed_scenarios)
        adversarial = AdversarialAgent("adversarial-agent-1", self.policy.adversarial_allowed_scenarios)
        return [normal, adversarial]

    def run(
        self,
        turns: Optional[int] = None,
        agents: Optional[List[Agent]] = None,
        session_id: Optional[str] = None,
        perturbation: Optional[float] = None,
    ) -> List[TurnResult]:
        """
        Runs `turns` agent actions. All turns in one run() call share the
        same session_id by default (so repeated adversarial turns
        correlate, per experiment D) -- pass a fresh session_id per call
        for independent sessions.
        """
        turns = turns if turns is not None else self.policy.default_turns
        agents = agents if agents is not None else self.build_default_agents()
        session_id = session_id or f"sess-{uuid.uuid4().hex[:10]}"
        magnitude = self.policy.perturbation_default if perturbation is None else perturbation
        if not (self.policy.perturbation_min <= magnitude <= self.policy.perturbation_max):
            raise ValueError(
                f"perturbation {magnitude} outside configured bounds "
                f"[{self.policy.perturbation_min}, {self.policy.perturbation_max}]"
            )

        results: List[TurnResult] = []
        for turn_index in range(turns):
            agent = agents[turn_index % len(agents)]
            action = agent.act(session_id, self.rng)
            result = self._execute_turn(action, turn_index, magnitude)
            results.append(result)
        return results

    def _execute_turn(self, action: AgentAction, turn_index: int, magnitude: float) -> TurnResult:
        scenario = self.catalog.get(action.scenario_name)

        raw_sample, source_index, source_label, perturbed_fields = generate_sample(
            scenario, self.exemplar_bank, magnitude, self.rng
        )

        sample_id = f"{action.session_id}{SESSION_DELIMITER}{turn_index}{SESSION_DELIMITER}{uuid.uuid4().hex[:8]}"

        record = GeneratedTrafficRecord(
            sample_id=sample_id, session_id=action.session_id, agent_id=action.agent_id,
            agent_type=action.agent_type, scenario_name=scenario.name, intended_label=scenario.category,
            source_exemplar_index=source_index, source_exemplar_label=source_label,
            perturbation_magnitude=magnitude, perturbed_fields=perturbed_fields, timestamp=_now_iso(),
        )
        self.records.append(record)

        log_event(
            logger, 20, "agent traffic generated", correlation_id=action.session_id,
            sample_id=sample_id, agent_id=action.agent_id, agent_type=action.agent_type,
            scenario=scenario.name, intended_label=scenario.category,
            source_exemplar_index=source_index, source_exemplar_label=source_label,
            perturbation_magnitude=magnitude, perturbed_field_count=len(perturbed_fields),
        )

        pipeline_message = self.gateway.ingest(sample_id, raw_sample)

        return TurnResult(sample_id=sample_id, session_id=action.session_id, raw_sample=raw_sample, record=record, pipeline_message=pipeline_message)
