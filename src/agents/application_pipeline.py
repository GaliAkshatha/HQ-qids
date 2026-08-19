"""
src/agents/application_pipeline.py

Wires the application-security detector through the EXISTING
routing -> hybrid -> risk -> defense -> incident architecture (Phases
3-6, completely unmodified) using NEW, application-specific artifacts --
never the NSL-KDD-trained objects.

IncidentManager.record_full_lifecycle() (the exact extension point built
for Phase 7's distributed workers) is reused directly here: it accepts
pre-computed contract objects without internally calling the NSL-KDD-
specific transform_sample() the way IncidentManager.process() does --
making it the correct integration seam for a second, parallel detection
source.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.agents.application_detector import ApplicationSecurityDetector
from src.agents.application_features import ApplicationFeatureVector
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.hybrid.pipeline import HybridPipeline
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import EventStore
from src.incident.incident_manager import IncidentManager
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

SOURCE_LABEL = "APPLICATION_SECURITY"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApplicationPipelineResult:
    incident_snapshot: object
    detection_source: str = SOURCE_LABEL


class ApplicationSecurityPipeline:
    def __init__(
        self,
        detector: ApplicationSecurityDetector,
        quantum_verifier,
        event_store: EventStore,
        routing_policy: Optional[RoutingPolicyConfig] = None,
        defense_policy: Optional[DefensePolicyConfig] = None,
        escalation_policy: Optional[EscalationPolicyConfig] = None,
    ) -> None:
        self.detector = detector
        self.routing_policy = routing_policy or RoutingPolicyConfig.load()
        self.router = QuantumRouter(policy=self.routing_policy, verifier=quantum_verifier, job_queue=QuantumJobQueue(max_workers=2))
        self.hybrid_pipeline = HybridPipeline()
        self.defense_engine = DefenseEngine(policy=defense_policy or DefensePolicyConfig.load())
        self.incident_manager = IncidentManager(
            detector=None, router=None, hybrid_pipeline=None, defense_engine=None,
            event_store=event_store, escalation_policy=escalation_policy or EscalationPolicyConfig.load(),
        )

    def process(self, feature_vector: ApplicationFeatureVector, correlation_key: str) -> ApplicationPipelineResult:
        sample_id = f"app-{feature_vector.session_id}-{uuid.uuid4().hex[:8]}"
        incident_id = str(uuid.uuid4())
        created_at = _now_iso()

        detection_result = self.detector.detect(feature_vector, sample_id=sample_id)

        scaled = np.array([[getattr(feature_vector, name) for name in self.detector.feature_names]])
        scaled = self.detector.scaler.transform(scaled)[0]
        routing_decision = self.router.route_and_wait(sample_id, scaled, detection_result, timeout=10)

        hybrid_decision, risk_assessment = self.hybrid_pipeline.process(detection_result, routing_decision)

        rollback_before = self.defense_engine.metrics.snapshot()["rollbacks"]
        defense_result = self.defense_engine.process(detection_result, hybrid_decision, risk_assessment)
        rollback_occurred = self.defense_engine.metrics.snapshot()["rollbacks"] > rollback_before

        snapshot = self.incident_manager.record_full_lifecycle(
            correlation_key=correlation_key, incident_id=incident_id, created_at=created_at,
            detection_result=detection_result, routing_decision=routing_decision,
            hybrid_decision=hybrid_decision, risk_assessment=risk_assessment, defense_result=defense_result,
            rollback_occurred=rollback_occurred,
        )

        return ApplicationPipelineResult(incident_snapshot=snapshot)
