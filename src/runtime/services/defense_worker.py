"""
src/runtime/services/defense_worker.py

Consumes risk.assessed, calls DefenseEngine.process() directly (Phase 5,
unmodified), publishes defense.completed. Also computes rollback_occurred
via the same before/after DefenseEngine.metrics diff technique Phase 6's
single-process IncidentManager used -- and includes it explicitly in the
payload, since the distributed incident-worker has no access to this
worker's DefenseEngine instance to compute it independently.

Defense side-effect exactly-once is enforced by TWO independent layers:
Phase 5's own safety-validator already-active check (domain-level), plus
this worker's Redis event-id idempotency check (transport-level) --
neither replaces the other.
"""

from __future__ import annotations

from typing import Optional

import redis

from src.contracts import DetectionResult, HybridDecision, PipelineMessage, RiskAssessment
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.message_helpers import new_message, routing_decision_from_dict
from src.runtime.redis_client import build_redis_client, check_redis_connectivity
from src.runtime.stream_worker import StreamWorker


class DefenseWorker(StreamWorker):
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, defense_engine: DefenseEngine) -> None:
        self.defense_engine = defense_engine
        self.config = config
        logger = get_logger("defense_worker")
        idem = RedisIdempotencyStore(client, config, "defense_worker")
        super().__init__(
            "defense-worker", client, config, config.streams["risk_assessed"],
            config.consumer_groups["defense_worker"], idem, logger,
        )

    def output_stream(self) -> Optional[str]:
        return self.config.streams["defense_completed"]

    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        detection_result = DetectionResult(**message.payload["detection_result"])
        routing_decision = routing_decision_from_dict(message.payload["routing_decision"])
        hybrid_decision = HybridDecision(**message.payload["hybrid_decision"])
        risk_assessment = RiskAssessment(**message.payload["risk_assessment"])

        rollback_count_before = self.defense_engine.metrics.snapshot()["rollbacks"]
        defense_result = self.defense_engine.process(detection_result, hybrid_decision, risk_assessment)
        rollback_occurred = self.defense_engine.metrics.snapshot()["rollbacks"] > rollback_count_before

        return new_message(
            "defense.completed", message.correlation_id, message.incident_id,
            {
                "sample_id": message.payload["sample_id"], "raw_sample": message.payload["raw_sample"],
                "detection_result": detection_result.to_dict(), "routing_decision": routing_decision.to_dict(),
                "hybrid_decision": hybrid_decision.to_dict(), "risk_assessment": risk_assessment.to_dict(),
                "defense_result": defense_result.to_dict(), "rollback_occurred": rollback_occurred,
            },
            causation_id=message.event_id,
        )


def build_worker(config: Optional[RuntimePolicyConfig] = None) -> DefenseWorker:
    config = config or RuntimePolicyConfig.load()
    client = build_redis_client(config)
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load())
    return DefenseWorker(client, config, defense_engine)


def readiness_checks(config: RuntimePolicyConfig, client: redis.Redis):
    return {"redis": lambda: check_redis_connectivity(client)}


if __name__ == "__main__":
    cfg = RuntimePolicyConfig.load()
    worker = build_worker(cfg)
    app = build_health_app("defense-worker", readiness_checks(cfg, worker.client))
    run_health_server_in_background(app, cfg.health_ports["defense_worker_port"])
    worker.run_forever()
