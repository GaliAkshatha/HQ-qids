"""
src/hybrid/metrics.py

In-process, thread-safe counters for the hybrid decision + risk layer.
Mirrors src/routing/metrics.py's RouterMetrics pattern. Descriptive only
-- these metrics measure what happened, they do not claim quantum
verification improved anything (that requires a held-out evaluation,
explicitly out of scope for Phase 4).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict

from src.contracts import HybridDecision, RiskAssessment


@dataclass
class HybridMetrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    total_samples: int = 0
    classical_predictions: Dict[str, int] = field(default_factory=dict)
    final_predictions: Dict[str, int] = field(default_factory=dict)
    risk_distribution: Dict[str, int] = field(default_factory=dict)

    quantum_invocations: int = 0  # quantum_used True OR fallback (i.e. was attempted)
    quantum_successes: int = 0  # decision_status == "success" upstream (agreement or disagreement, resolved)
    quantum_failures: int = 0  # fallback_used True
    classical_fallbacks: int = 0  # alias of quantum_failures, kept separate for clarity in reporting

    agreements: int = 0
    disagreements: int = 0
    decision_changes: int = 0

    def record(self, classical_prediction: str, hybrid_decision: HybridDecision, risk: RiskAssessment) -> None:
        with self._lock:
            self.total_samples += 1
            self.classical_predictions[classical_prediction] = self.classical_predictions.get(classical_prediction, 0) + 1
            self.final_predictions[hybrid_decision.final_prediction] = (
                self.final_predictions.get(hybrid_decision.final_prediction, 0) + 1
            )
            self.risk_distribution[risk.risk_level] = self.risk_distribution.get(risk.risk_level, 0) + 1

            ev = hybrid_decision.evidence
            fallback_used = bool(ev.get("fallback_used"))
            agreement = ev.get("agreement")

            if hybrid_decision.quantum_used or fallback_used:
                self.quantum_invocations += 1

            if fallback_used:
                self.quantum_failures += 1
                self.classical_fallbacks += 1

            if hybrid_decision.quantum_used and agreement is not None:
                self.quantum_successes += 1
                if agreement:
                    self.agreements += 1
                else:
                    self.disagreements += 1
                if ev.get("decision_changed_by_quantum"):
                    self.decision_changes += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            def _rate(numerator: int, denominator: int):
                return numerator / denominator if denominator else None

            return {
                "total_samples": self.total_samples,
                "classical_predictions": dict(self.classical_predictions),
                "final_predictions": dict(self.final_predictions),
                "risk_distribution": dict(self.risk_distribution),
                "quantum_invocations": self.quantum_invocations,
                "quantum_successes": self.quantum_successes,
                "quantum_failures": self.quantum_failures,
                "classical_fallbacks": self.classical_fallbacks,
                "agreements": self.agreements,
                "disagreements": self.disagreements,
                "decision_changes": self.decision_changes,
                "quantum_confirmation_rate": _rate(self.agreements, self.quantum_successes),
                "quantum_disagreement_rate": _rate(self.disagreements, self.quantum_successes),
                "quantum_decision_change_rate": _rate(self.decision_changes, self.quantum_successes),
            }
