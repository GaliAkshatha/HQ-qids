"""
src/agents/metrics.py

Computes the required Phase 8 experiments from real TurnResult +
TurnOutcome pairs collected during a real run. No fabricated values --
every number here is derived from actually-observed IDS outcomes for
actually-generated samples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AgentRunMetrics:
    rows: List[Dict[str, object]] = field(default_factory=list)

    def record(
        self,
        intended_label: str,
        scenario_name: str,
        final_prediction: Optional[str],
        escalated: Optional[bool],
        selected_action: Optional[str],
        perturbation_magnitude: float,
        agent_type: str,
        session_id: str,
    ) -> None:
        self.rows.append({
            "intended_label": intended_label,
            "scenario_name": scenario_name,
            "final_prediction": final_prediction,
            "escalated": bool(escalated) if escalated is not None else None,
            "selected_action": selected_action,
            "perturbation_magnitude": perturbation_magnitude,
            "agent_type": agent_type,
            "session_id": session_id,
        })

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {
            "normal": {"normal": 0, "attack": 0, "unknown": 0},
            "attack": {"normal": 0, "attack": 0, "unknown": 0},
        }
        for row in self.rows:
            intended = row["intended_label"]
            predicted = row["final_prediction"] or "unknown"
            if intended not in matrix:
                continue
            matrix[intended][predicted] = matrix[intended].get(predicted, 0) + 1
        return matrix

    def confusion_matrix_accuracy(self) -> Optional[float]:
        matrix = self.confusion_matrix()
        correct = matrix["normal"]["normal"] + matrix["attack"]["attack"]
        total = sum(sum(v.values()) for v in matrix.values())
        return correct / total if total else None

    def escalation_rate_by_scenario(self) -> Dict[str, Optional[float]]:
        by_scenario: Dict[str, List[bool]] = defaultdict(list)
        for row in self.rows:
            if row["escalated"] is not None:
                by_scenario[row["scenario_name"]].append(row["escalated"])
        return {
            scenario: (sum(vals) / len(vals) if vals else None)
            for scenario, vals in by_scenario.items()
        }

    def defense_action_distribution_by_scenario(self) -> Dict[str, Dict[str, int]]:
        by_scenario: Dict[str, Counter] = defaultdict(Counter)
        for row in self.rows:
            if row["selected_action"]:
                by_scenario[row["scenario_name"]][row["selected_action"]] += 1
        return {scenario: dict(counter) for scenario, counter in by_scenario.items()}

    def scenario_distribution(self) -> Dict[str, int]:
        return dict(Counter(row["scenario_name"] for row in self.rows))

    def accuracy_by_perturbation_magnitude(self) -> Dict[float, Optional[float]]:
        by_magnitude: Dict[float, List[Tuple[str, Optional[str]]]] = defaultdict(list)
        for row in self.rows:
            by_magnitude[row["perturbation_magnitude"]].append((row["intended_label"], row["final_prediction"]))
        result: Dict[float, Optional[float]] = {}
        for magnitude, pairs in by_magnitude.items():
            observed = [(i, p) for i, p in pairs if p is not None]
            if not observed:
                result[magnitude] = None
                continue
            correct = sum(1 for intended, predicted in observed if intended == predicted)
            result[magnitude] = correct / len(observed)
        return result

    def total_samples(self) -> int:
        return len(self.rows)

    def samples_with_observed_outcome(self) -> int:
        return sum(1 for row in self.rows if row["final_prediction"] is not None)
