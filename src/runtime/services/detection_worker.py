"""
src/runtime/services/detection_worker.py

Consumes traffic.ingested, calls EnsembleClassicalDetector.detect()
directly (Phase 1, unmodified), publishes detection.completed. No
detection logic of its own -- pure translation between PipelineMessage
and DetectionResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import redis

from src.contracts import PipelineMessage
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.observability.logging_config import get_logger
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.idempotency import RedisIdempotencyStore
from src.runtime.message_helpers import new_message
from src.runtime.redis_client import build_redis_client, check_redis_connectivity
from src.runtime.stream_worker import StreamWorker

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSICAL_MODELS_DIR = REPO_ROOT / "artifacts" / "models" / "classical"
PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing"


class DetectionWorker(StreamWorker):
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, detector: EnsembleClassicalDetector) -> None:
        self.detector = detector
        self.config = config
        logger = get_logger("detection_worker")
        idem = RedisIdempotencyStore(client, config, "detection_worker")
        super().__init__(
            "detection-worker", client, config, config.streams["traffic_ingested"],
            config.consumer_groups["detection_worker"], idem, logger,
        )

    def output_stream(self) -> Optional[str]:
        return self.config.streams["detection_completed"]

    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        sample_id = message.payload["sample_id"]
        raw_sample = message.payload["raw_sample"]
        detection_result = self.detector.detect(raw_sample, sample_id=sample_id)
        return new_message(
            "detection.completed", message.correlation_id, message.incident_id,
            {"sample_id": sample_id, "raw_sample": raw_sample, "detection_result": detection_result.to_dict()},
            causation_id=message.event_id,
        )


def build_worker(config: Optional[RuntimePolicyConfig] = None) -> DetectionWorker:
    config = config or RuntimePolicyConfig.load()
    client = build_redis_client(config)
    detector = EnsembleClassicalDetector.load(models_dir=CLASSICAL_MODELS_DIR, preprocessing_dir=PREPROCESSING_DIR)
    return DetectionWorker(client, config, detector)


def readiness_checks(config: RuntimePolicyConfig, client: redis.Redis):
    return {
        "redis": lambda: check_redis_connectivity(client),
        "classical_model_artifacts": lambda: (CLASSICAL_MODELS_DIR / "random_forest.joblib").exists(),
        "preprocessing_artifacts": lambda: (PREPROCESSING_DIR / "scaler.joblib").exists(),
    }


if __name__ == "__main__":
    cfg = RuntimePolicyConfig.load()
    worker = build_worker(cfg)
    app = build_health_app("detection-worker", readiness_checks(cfg, worker.client))
    run_health_server_in_background(app, cfg.health_ports["detection_worker_port"])
    worker.run_forever()
