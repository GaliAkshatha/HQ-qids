"""
src/runtime/services/risk_worker.py

Consumes quantum.completed, calls HybridPipeline.process() directly
(Phase 4, unmodified), publishes risk.assessed carrying both
HybridDecision and RiskAssessment -- one stream, not two, since
HybridPipeline already produces both in a single call and there is no
separate lifecycle state for them (see the approved transition table).
"""

from __future__ import annotations

from typing import Optional

import redis

from src.contracts import DetectionResult, PipelineMessage
from src.hybrid.pipeline import HybridPipeline
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.message_helpers import new_message, routing_decision_from_dict
from src.runtime.redis_client import build_redis_client, check_redis_connectivity
from src.runtime.stream_worker import StreamWorker


class RiskWorker(StreamWorker):
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, hybrid_pipeline: HybridPipeline) -> None:
        self.hybrid_pipeline = hybrid_pipeline
        self.config = config
        logger = get_logger("risk_worker")
        idem = RedisIdempotencyStore(client, config, "risk_worker")
        super().__init__(
            "risk-worker", client, config, config.streams["quantum_completed"],
            config.consumer_groups["risk_worker"], idem, logger,
        )

    def output_stream(self) -> Optional[str]:
        return self.config.streams["risk_assessed"]

    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        detection_result = DetectionResult(**message.payload["detection_result"])
        routing_decision = routing_decision_from_dict(message.payload["routing_decision"])

        hybrid_decision, risk_assessment = self.hybrid_pipeline.process(detection_result, routing_decision)

        return new_message(
            "risk.assessed", message.correlation_id, message.incident_id,
            {
                "sample_id": message.payload["sample_id"], "raw_sample": message.payload["raw_sample"],
                "detection_result": detection_result.to_dict(), "routing_decision": routing_decision.to_dict(),
                "hybrid_decision": hybrid_decision.to_dict(), "risk_assessment": risk_assessment.to_dict(),
            },
            causation_id=message.event_id,
        )


def build_worker(config: Optional[RuntimePolicyConfig] = None) -> RiskWorker:
    config = config or RuntimePolicyConfig.load()
    client = build_redis_client(config)
    hybrid_pipeline = HybridPipeline()
    return RiskWorker(client, config, hybrid_pipeline)


def readiness_checks(config: RuntimePolicyConfig, client: redis.Redis):
    return {"redis": lambda: check_redis_connectivity(client)}


if __name__ == "__main__":
    cfg = RuntimePolicyConfig.load()
    worker = build_worker(cfg)
    app = build_health_app("risk-worker", readiness_checks(cfg, worker.client))
    run_health_server_in_background(app, cfg.health_ports["risk_worker_port"])
    worker.run_forever()
