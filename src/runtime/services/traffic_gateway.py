"""
src/runtime/services/traffic_gateway.py

Entry point of the distributed pipeline. Assigns correlation_id (via the
same SampleIdCorrelation logic Phase 6 uses, reused not reimplemented)
and incident_id for a sample's entire journey, then publishes
traffic.ingested. Not a StreamWorker -- it has nothing to consume from.

Phase 8 addition: correlation_strategy is now an optional constructor
parameter, defaulting to SampleIdCorrelation() exactly as before -- every
existing caller (Phases 1-7's own code and tests) is unaffected. This is
what lets src/agents/session_correlation.py's AgentSessionCorrelation be
used here (duck-typed, not a formal subclass -- see that module's
docstring) without any further change to this file.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

import redis

from src.incident.correlation import SampleIdCorrelation
from src.observability.logging_config import get_logger, log_event
from src.runtime.config import RuntimePolicyConfig
from src.runtime.health import build_health_app, run_health_server_in_background
from src.runtime.message_helpers import new_message
from src.runtime.redis_client import build_redis_client, check_redis_connectivity


class TrafficGateway:
    def __init__(self, client: redis.Redis, config: RuntimePolicyConfig, correlation_strategy: Optional[Any] = None) -> None:
        self.client = client
        self.config = config
        self.correlation_strategy = correlation_strategy or SampleIdCorrelation()
        self.logger = get_logger("traffic_gateway")

    def ingest(self, sample_id: str, raw_sample: dict):
        correlation_id = self.correlation_strategy.correlation_key(sample_id)
        incident_id = str(uuid.uuid4())
        message = new_message(
            "traffic.ingested", correlation_id, incident_id,
            {"sample_id": sample_id, "raw_sample": raw_sample},
            causation_id=None,
        )
        self.client.xadd(self.config.streams["traffic_ingested"], {"data": json.dumps(message.to_dict(), default=str)})
        log_event(
            self.logger, 20, "traffic ingested", correlation_id=correlation_id, incident_id=incident_id,
            causation_id=None, event_type="traffic.ingested", event_id=message.event_id,
        )
        return message


def start_health_server(config: RuntimePolicyConfig, client: redis.Redis):
    app = build_health_app("traffic-gateway", {"redis": lambda: check_redis_connectivity(client)})
    return run_health_server_in_background(app, config.health_ports["traffic_gateway_port"])


if __name__ == "__main__":
    import time

    cfg = RuntimePolicyConfig.load()
    redis_client = build_redis_client(cfg)
    start_health_server(cfg, redis_client)
    print("traffic-gateway running (health server started); use TrafficGateway.ingest() to submit samples")
    while True:
        time.sleep(3600)
