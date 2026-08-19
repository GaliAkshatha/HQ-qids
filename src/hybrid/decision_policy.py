"""
src/hybrid/decision_policy.py

Loads config/hybrid_decision_policy.json. Nothing in the hybrid engine
hard-codes a threshold -- the 0.85 override bar and 0.90 classical-confirm
bar are both initial, principled-but-arbitrary starting points, not tuned
against the test set (same discipline as Phase 3's routing thresholds).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "hybrid_decision_policy.json"


@dataclass
class DecisionPolicyConfig:
    quantum_override_confidence_threshold: float
    classical_high_confidence_threshold: float

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "DecisionPolicyConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            quantum_override_confidence_threshold=raw["quantum_override_confidence_threshold"],
            classical_high_confidence_threshold=raw["classical_high_confidence_threshold"],
        )
