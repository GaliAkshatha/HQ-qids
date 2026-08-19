"""
src/hybrid/risk_policy.py

Loads config/risk_policy.json. Validates that each weight group sums to
1.0 at load time -- a misconfigured policy fails loudly here rather than
silently producing a risk_score outside [0,1] later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "risk_policy.json"

_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass
class RiskPolicyConfig:
    combination_weights: Dict[str, float]
    threat_evidence_weights: Dict[str, float]
    system_uncertainty_weights: Dict[str, float]
    low_max: float
    medium_max: float
    high_max: float
    confirmed_attack_min_level: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "RiskPolicyConfig":
        with open(path) as f:
            raw = json.load(f)

        combination_weights = raw["combination_weights"]
        threat_weights = raw["threat_evidence_weights"]
        uncertainty_weights = raw["system_uncertainty_weights"]

        _validate_weight_sum(combination_weights, "combination_weights")
        _validate_weight_sum(threat_weights, "threat_evidence_weights")
        _validate_weight_sum(uncertainty_weights, "system_uncertainty_weights")

        thresholds = raw["thresholds"]
        return cls(
            combination_weights=combination_weights,
            threat_evidence_weights=threat_weights,
            system_uncertainty_weights=uncertainty_weights,
            low_max=thresholds["low_max"],
            medium_max=thresholds["medium_max"],
            high_max=thresholds["high_max"],
            confirmed_attack_min_level=raw["floors"]["confirmed_attack_min_level"],
        )


def _validate_weight_sum(weights: Dict[str, float], name: str) -> None:
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1.0, got {total} ({weights})")
