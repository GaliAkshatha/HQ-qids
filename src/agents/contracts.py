"""
src/agents/contracts.py

Agent-local data shapes. Deliberately kept OUT of src/contracts/ (the
shared surface consumed by the core IDS domain modules) even though this
project's established convention is to put typed interchange objects
there -- putting these here instead makes the "core is unaware of
agents" boundary visible in the source tree, not just true in practice.
Nothing under src/detection, src/quantum, src/routing, src/hybrid,
src/defense, or src/incident imports anything from this module.

These are PURE DATA. No agent here can perform any network, socket, or
subprocess operation -- there is no method on any of these dataclasses
that does anything beyond hold values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    category: str  # "normal" | "attack" -- the intended ground-truth label
    source_label: str  # the real NSL-KDD label this scenario samples exemplars from
    description: str
    default_perturbation: float

    def __post_init__(self):
        if self.category not in ("normal", "attack"):
            raise ValueError(f"category must be 'normal' or 'attack'. Received: {self.category}")
        if not 0.0 <= self.default_perturbation <= 1.0:
            raise ValueError(f"default_perturbation must be in [0,1]. Received: {self.default_perturbation}")


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    agent_type: str  # "normal" | "adversarial" | "environment"
    allowed_scenarios: tuple  # tuple, not list -- frozen/hashable, and immutable by construction

    def __post_init__(self):
        if self.agent_type not in ("normal", "adversarial", "environment"):
            raise ValueError(f"agent_type must be 'normal', 'adversarial', or 'environment'. Received: {self.agent_type}")
        if not self.allowed_scenarios:
            raise ValueError("allowed_scenarios must be non-empty")


@dataclass
class GeneratedTrafficRecord:
    """
    Full provenance for one generated sample -- logged for every sample,
    per the requirement to record source exemplar, scenario, agent,
    session, perturbation parameters, and intended ground-truth label.
    """

    sample_id: str
    session_id: str
    agent_id: str
    agent_type: str
    scenario_name: str
    intended_label: str  # "normal" | "attack" -- the ground truth this sample was generated to represent
    source_exemplar_index: int  # row index in KDDTrain+.txt this sample was derived from
    source_exemplar_label: str  # the real NSL-KDD label of that source row
    perturbation_magnitude: float
    perturbed_fields: Dict[str, Any] = field(default_factory=dict)  # {field: {"before":..., "after":...}}
    timestamp: Optional[str] = None
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
