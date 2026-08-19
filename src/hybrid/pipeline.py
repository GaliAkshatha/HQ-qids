"""
src/hybrid/pipeline.py

Convenience orchestrator: loads both policies once, and wraps
build_hybrid_decision() + assess_risk() + metrics recording into a single
call. Not new architecture -- just glue, mirroring how QuantumRouter
already bundles policy + execution + metrics for Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from src.contracts import DetectionResult, HybridDecision, RiskAssessment, RoutingDecision
from src.hybrid.decision_policy import DecisionPolicyConfig
from src.hybrid.hybrid_engine import build_hybrid_decision
from src.hybrid.metrics import HybridMetrics
from src.hybrid.risk_engine import assess_risk
from src.hybrid.risk_policy import RiskPolicyConfig


class HybridPipeline:
    def __init__(
        self,
        decision_policy: Optional[DecisionPolicyConfig] = None,
        risk_policy: Optional[RiskPolicyConfig] = None,
        metrics: Optional[HybridMetrics] = None,
    ) -> None:
        self.decision_policy = decision_policy or DecisionPolicyConfig.load()
        self.risk_policy = risk_policy or RiskPolicyConfig.load()
        self.metrics = metrics or HybridMetrics()

    def process(
        self, detection_result: DetectionResult, routing_decision: RoutingDecision
    ) -> Tuple[HybridDecision, RiskAssessment]:
        hybrid_decision = build_hybrid_decision(detection_result, routing_decision, self.decision_policy)
        risk = assess_risk(detection_result, hybrid_decision, self.risk_policy)
        self.metrics.record(detection_result.classical_prediction, hybrid_decision, risk)
        return hybrid_decision, risk

    def metrics_snapshot(self):
        return self.metrics.snapshot()
