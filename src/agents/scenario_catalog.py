"""
src/agents/scenario_catalog.py

Loads config/agent_scenarios.json and config/agent_environment_policy.json.
Malformed configuration fails loudly at load time with a clear message,
not silently later during generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from src.agents.contracts import ScenarioDefinition

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_PATH = REPO_ROOT / "config" / "agent_scenarios.json"
DEFAULT_ENV_POLICY_PATH = REPO_ROOT / "config" / "agent_environment_policy.json"


class ScenarioCatalogError(ValueError):
    pass


class ScenarioCatalog:
    def __init__(self, scenarios: Dict[str, ScenarioDefinition]) -> None:
        self.scenarios = scenarios

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCENARIOS_PATH) -> "ScenarioCatalog":
        with open(path) as f:
            raw = json.load(f)

        scenarios_raw = raw.get("scenarios")
        if not scenarios_raw:
            raise ScenarioCatalogError(f"{path}: missing or empty 'scenarios' key")

        scenarios: Dict[str, ScenarioDefinition] = {}
        for name, spec in scenarios_raw.items():
            try:
                scenarios[name] = ScenarioDefinition(
                    name=name,
                    category=spec["category"],
                    source_label=spec["source_label"],
                    description=spec["description"],
                    default_perturbation=spec["default_perturbation"],
                )
            except KeyError as e:
                raise ScenarioCatalogError(f"scenario '{name}' is missing required field: {e}") from e
            except ValueError as e:
                raise ScenarioCatalogError(f"scenario '{name}' is invalid: {e}") from e

        return cls(scenarios)

    def get(self, name: str) -> ScenarioDefinition:
        if name not in self.scenarios:
            raise ScenarioCatalogError(f"Unknown scenario: '{name}'. Known scenarios: {sorted(self.scenarios)}")
        return self.scenarios[name]

    def names(self) -> List[str]:
        return list(self.scenarios.keys())


@dataclass
class EnvironmentPolicy:
    default_turns: int
    normal_agent_weight: float
    adversarial_agent_weight: float
    normal_allowed_scenarios: Tuple[str, ...]
    adversarial_allowed_scenarios: Tuple[str, ...]
    perturbation_min: float
    perturbation_max: float
    perturbation_default: float
    correlation_strategy: str
    feedback_poll_timeout_ms: int
    feedback_poll_block_ms: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ENV_POLICY_PATH) -> "EnvironmentPolicy":
        with open(path) as f:
            raw = json.load(f)
        try:
            run = raw["run"]
            normal = raw["normal"]
            adversarial = raw["adversarial"]
            perturbation = raw["perturbation"]
            correlation = raw["correlation"]
            feedback = raw["feedback"]
            return cls(
                default_turns=run["default_turns"],
                normal_agent_weight=run["normal_agent_weight"],
                adversarial_agent_weight=run["adversarial_agent_weight"],
                normal_allowed_scenarios=tuple(normal["allowed_scenarios"]),
                adversarial_allowed_scenarios=tuple(adversarial["allowed_scenarios"]),
                perturbation_min=perturbation["min"],
                perturbation_max=perturbation["max"],
                perturbation_default=perturbation["default"],
                correlation_strategy=correlation["strategy"],
                feedback_poll_timeout_ms=feedback["poll_timeout_ms"],
                feedback_poll_block_ms=feedback["poll_block_ms"],
            )
        except KeyError as e:
            raise ScenarioCatalogError(f"{path}: missing required config field: {e}") from e
