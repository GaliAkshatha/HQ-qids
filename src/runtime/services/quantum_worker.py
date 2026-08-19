"""
src/runtime/services/quantum_worker.py

Consumes detection.completed, calls QuantumRouter.route_and_wait()
directly (Phase 3, unmodified), publishes quantum.completed.

IMPORTANT SEMANTIC: quantum.completed means "the quantum-routing/
verification STAGE has completed" -- it does NOT mean quantum computation
occurred. The full RoutingDecision (including should_invoke_quantum,
decision_status, fallback_reason, and quantum_result which is itself None
when not invoked) is preserved in the payload so downstream workers can
distinguish invoked-success / intentionally-skipped / circuit-open /
attempted-and-failed / classical-fallback. No worker infers quantum usage
from the event_type string alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import redis

from src.contracts import DetectionResult, PipelineMessage
from src.observability.logging_config import get_logger
from src.preprocessing.classical_pipeline import load_preprocessing_artifacts, transform_sample
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.message_helpers import new_message
from src.runtime.redis_client import build_redis_client, check_redis_connectivity
from src.runtime.stream_worker import StreamWorker

REPO_ROOT = Path(__file__).resolve().parents[3]
VQC_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "quantum" / "vqc"
PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"


class QuantumWorker(StreamWorker):
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, router: QuantumRouter, preprocessing_artifacts) -> None:
        self.router = router
        self.preprocessing_artifacts = preprocessing_artifacts
        self.config = config
        logger = get_logger("quantum_worker")
        idem = RedisIdempotencyStore(client, config, "quantum_worker")
        super().__init__(
            "quantum-worker", client, config, config.streams["detection_completed"],
            config.consumer_groups["quantum_worker"], idem, logger,
        )

    def output_stream(self) -> Optional[str]:
        return self.config.streams["quantum_completed"]

    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        sample_id = message.payload["sample_id"]
        raw_sample = message.payload["raw_sample"]
        detection_result = DetectionResult(**message.payload["detection_result"])

        scaled_features = transform_sample(raw_sample, self.preprocessing_artifacts)[0]
        routing_decision = self.router.route_and_wait(sample_id, scaled_features, detection_result, timeout=10)

        return new_message(
            "quantum.completed", message.correlation_id, message.incident_id,
            {
                "sample_id": sample_id, "raw_sample": raw_sample,
                "detection_result": detection_result.to_dict(),
                "routing_decision": routing_decision.to_dict(),
            },
            causation_id=message.event_id,
        )


def build_worker(config: Optional[RuntimePolicyConfig] = None) -> QuantumWorker:
    config = config or RuntimePolicyConfig.load()
    client = build_redis_client(config)
    routing_policy = RoutingPolicyConfig.load()
    verifier = VQCVerifier.load(models_dir=VQC_MODELS_DIR, preprocessing_dir=PREPROCESSING_DIR)
    router = QuantumRouter(policy=routing_policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))
    preprocessing_artifacts = load_preprocessing_artifacts(PREPROCESSING_DIR)
    return QuantumWorker(client, config, router, preprocessing_artifacts)


def readiness_checks(config: RuntimePolicyConfig, client: redis.Redis):
    return {
        "redis": lambda: check_redis_connectivity(client),
        "vqc_model_artifacts": lambda: (VQC_MODELS_DIR / "vqc_model.dill").exists(),
        "quantum_pca_artifact": lambda: (PREPROCESSING_DIR / "quantum_pca.joblib").exists(),
        # Deliberately does NOT run a real inference call -- artifact
        # presence is sufficient for readiness, per the approved plan.
    }


if __name__ == "__main__":
    cfg = RuntimePolicyConfig.load()
    worker = build_worker(cfg)
    app = build_health_app("quantum-worker", readiness_checks(cfg, worker.client))
    run_health_server_in_background(app, cfg.health_ports["quantum_worker_port"])
    worker.run_forever()
