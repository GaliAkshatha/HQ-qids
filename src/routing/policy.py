"""
src/routing/policy.py

Loads routing thresholds from config/routing_policy.json (outside src, per
approved Decision 2) -- nothing in this module hard-codes a threshold.
evaluate_routing() implements exactly the OR-of-thresholds pattern from
the approved spec: any signal past its threshold triggers quantum
verification, with the specific reason(s) recorded.

Defaults shipped in config/routing_policy.json (confidence 0.70, anomaly
0.70, disagreement 0.30) are initial, principled-but-arbitrary starting
points -- not claimed optimal, not tuned against the test set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.contracts import DetectionResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "routing_policy.json"


@dataclass
class RoutingPolicyConfig:
    confidence_threshold: float
    anomaly_threshold: float
    disagreement_threshold: float
    combination: str

    quantum_backend: str

    circuit_breaker_failure_threshold: int
    circuit_breaker_cooldown_seconds: float

    timeout_seconds: Dict[str, float]
    max_retries: int
    backoff_seconds: float

    queue_max_workers: int

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "RoutingPolicyConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "RoutingPolicyConfig":
        routing = raw["routing"]
        breaker = raw["circuit_breaker"]
        execution = raw["execution"]
        queue = raw["queue"]
        return cls(
            confidence_threshold=routing["confidence_threshold"],
            anomaly_threshold=routing["anomaly_threshold"],
            disagreement_threshold=routing["disagreement_threshold"],
            combination=routing["combination"],
            quantum_backend=raw["quantum_backend"],
            circuit_breaker_failure_threshold=breaker["failure_threshold"],
            circuit_breaker_cooldown_seconds=breaker["cooldown_seconds"],
            timeout_seconds=execution["timeout_seconds"],
            max_retries=execution["max_retries"],
            backoff_seconds=execution["backoff_seconds"],
            queue_max_workers=queue["max_workers"],
        )

    def with_overrides(self, **overrides) -> "RoutingPolicyConfig":
        """Return a copy with specific fields overridden -- used e.g. to
        point a measurement run at QSVM instead of the default VQC backend,
        without checking in a second config file for every variant."""
        import dataclasses

        return dataclasses.replace(self, **overrides)

    def threshold_snapshot(self) -> Dict[str, float]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "disagreement_threshold": self.disagreement_threshold,
        }

    def timeout_for(self, backend: str) -> float:
        if backend not in self.timeout_seconds:
            raise ValueError(f"No configured timeout_seconds for backend '{backend}'")
        return self.timeout_seconds[backend]


def evaluate_routing(
    detection_result: DetectionResult, policy: RoutingPolicyConfig
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Returns (should_invoke_quantum, reason_codes, signal_values).

    Signals used are exactly DetectionResult's three continuous fields --
    no signal is invented that isn't actually in the (unmodified) contract.
    """
    signal_values = {
        "classical_confidence": detection_result.classical_confidence,
        "anomaly_score": detection_result.anomaly_score,
        "model_disagreement": detection_result.model_disagreement,
    }

    reason_codes: List[str] = []
    if signal_values["classical_confidence"] < policy.confidence_threshold:
        reason_codes.append("LOW_CONFIDENCE")
    if signal_values["anomaly_score"] > policy.anomaly_threshold:
        reason_codes.append("HIGH_ANOMALY")
    if signal_values["model_disagreement"] > policy.disagreement_threshold:
        reason_codes.append("HIGH_DISAGREEMENT")

    if policy.combination == "any":
        should_invoke = len(reason_codes) > 0
    else:
        raise ValueError(f"Unsupported routing combination mode: '{policy.combination}'")

    return should_invoke, reason_codes, signal_values
