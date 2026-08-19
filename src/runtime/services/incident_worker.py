"""
src/runtime/services/incident_worker.py

Consumes defense.completed, calls IncidentManager.record_full_lifecycle()
directly (Phase 6, extended by the approved extraction) with the five
evidence objects it received via Redis -- it does NOT call
IncidentManager.process(), which would re-run detection/routing/hybrid/
defense a second time. Publishes incident.updated.

Incident-transition exactly-once uses TWO independent layers: Phase 6's
own terminal-incident check (IncidentManager.get_incident, keyed by
incident_id -- the true idempotency identity, domain-level) plus this
worker's Redis event-id idempotency check (transport-level) -- neither
replaces the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import redis

from src.contracts import DefenseResult, DetectionResult, HybridDecision, PipelineMessage, RiskAssessment
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import JsonlEventStore
from src.incident.incident_manager import IncidentManager
from src.observability.logging_config import get_logger, log_event
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.message_helpers import new_message, now_iso, routing_decision_from_dict
from src.runtime.redis_client import build_redis_client, check_redis_connectivity
from src.runtime.stream_worker import StreamWorker

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSICAL_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "classical"
VQC_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "quantum" / "vqc"
PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"
DEFAULT_EVENT_STORE_PATH = REPO_ROOT / "logs" / "distributed_incident_events.jsonl"


class IncidentWorker(StreamWorker):
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, incident_manager: IncidentManager) -> None:
        self.incident_manager = incident_manager
        self.config = config
        logger = get_logger("incident_worker")
        idem = RedisIdempotencyStore(client, config, "incident_worker")
        super().__init__(
            "incident-worker", client, config, config.streams["defense_completed"],
            config.consumer_groups["incident_worker"], idem, logger,
        )

    def output_stream(self) -> Optional[str]:
        return self.config.streams["incident_updated"]

    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        correlation_key = message.correlation_id
        incident_id = message.incident_id

        # Incident-level idempotency (domain layer, distinct from the
        # transport-layer event_id dedup StreamWorker already applied).
        # Keyed by incident_id -- the true idempotency identity -- NOT
        # correlation_key. A correlation_key may legitimately map to many
        # distinct incident_ids (e.g. Phase 8's AgentSessionCorrelation);
        # only a REDELIVERY of this exact incident_id should be skipped.
        existing = self.incident_manager.get_incident(incident_id)
        if existing is not None and existing.is_terminal:
            self.incident_manager.append_idempotent_skip(
                existing, "incident_id already processed to a terminal state -- distributed incident-worker skipping reprocessing"
            )
            log_event(
                self.logger, 20, "incident-level idempotent skip", correlation_id=correlation_key,
                incident_id=incident_id, event_type=message.event_type,
            )
            return new_message(
                "incident.updated", correlation_key, incident_id,
                {"current_state": existing.current_state, "skipped": True},
                causation_id=message.event_id,
            )

        detection_result = DetectionResult(**message.payload["detection_result"])
        routing_decision = routing_decision_from_dict(message.payload["routing_decision"])
        hybrid_decision = HybridDecision(**message.payload["hybrid_decision"])
        risk_assessment = RiskAssessment(**message.payload["risk_assessment"])
        defense_result = DefenseResult(**message.payload["defense_result"])
        rollback_occurred = message.payload["rollback_occurred"]

        snapshot = self.incident_manager.record_full_lifecycle(
            correlation_key=correlation_key, incident_id=incident_id, created_at=now_iso(),
            detection_result=detection_result, routing_decision=routing_decision,
            hybrid_decision=hybrid_decision, risk_assessment=risk_assessment, defense_result=defense_result,
            rollback_occurred=rollback_occurred,
        )

        return new_message(
            "incident.updated", correlation_key, incident_id,
            {"current_state": snapshot.current_state, "escalated": snapshot.escalated, "skipped": False},
            causation_id=message.event_id,
        )


def build_worker(config: Optional[RuntimePolicyConfig] = None, event_store_path: Optional[Path] = None) -> IncidentWorker:
    config = config or RuntimePolicyConfig.load()
    client = build_redis_client(config)

    detector = EnsembleClassicalDetector.load(models_dir=CLASSICAL_MODELS_DIR, preprocessing_dir=PREPROCESSING_DIR)
    routing_policy = RoutingPolicyConfig.load()
    verifier = VQCVerifier.load(models_dir=VQC_MODELS_DIR, preprocessing_dir=PREPROCESSING_DIR)
    router = QuantumRouter(policy=routing_policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=2))
    hybrid_pipeline = HybridPipeline()
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load())
    escalation_policy = EscalationPolicyConfig.load()
    event_store = JsonlEventStore(event_store_path or DEFAULT_EVENT_STORE_PATH)

    incident_manager = IncidentManager(
        detector=detector, router=router, hybrid_pipeline=hybrid_pipeline, defense_engine=defense_engine,
        event_store=event_store, escalation_policy=escalation_policy,
    )
    return IncidentWorker(client, config, incident_manager)


def readiness_checks(config: RuntimePolicyConfig, client: redis.Redis):
    return {"redis": lambda: check_redis_connectivity(client)}


if __name__ == "__main__":
    cfg = RuntimePolicyConfig.load()
    worker = build_worker(cfg)
    app = build_health_app("incident-worker", readiness_checks(cfg, worker.client))
    run_health_server_in_background(app, cfg.health_ports["incident_worker_port"])
    worker.run_forever()
