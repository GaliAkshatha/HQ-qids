"""
src/runtime/dead_letter.py

Publishes a permanently-failed PipelineMessage to the events.dead_letter
stream, with failure metadata attached -- the terminal outcome of the
retry-with-backoff mechanism in stream_worker.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import redis

from src.contracts import PipelineMessage
from src.runtime.config import RuntimePolicyConfig


def publish_to_dead_letter(
    client: redis.Redis,
    config: RuntimePolicyConfig,
    message: PipelineMessage,
    error: str,
    failed_stream: str,
    failed_worker: str,
) -> str:
    dlq_payload = {
        "message": json.dumps(message.to_dict(), default=str),
        "error": error,
        "failed_stream": failed_stream,
        "failed_worker": failed_worker,
        "retry_count": str(message.retry_count),
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    stream_id = client.xadd(config.streams["dead_letter"], dlq_payload)
    return stream_id
