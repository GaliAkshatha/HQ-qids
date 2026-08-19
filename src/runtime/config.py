"""
src/runtime/config.py

Loads config/runtime_policy.json. No hardcoded operational config
anywhere in src/runtime/ -- host/port/stream names/retry counts/etc. all
come from here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "config" / "runtime_policy.json"


@dataclass
class RuntimePolicyConfig:
    redis_host: str
    redis_port: int
    redis_db: int

    streams: Dict[str, str]
    consumer_groups: Dict[str, str]

    max_retries: int
    backoff_seconds: float
    backoff_multiplier: float

    min_idle_time_ms: int
    claim_batch_size: int

    idempotency_key_prefix: str
    idempotency_ttl_seconds: int

    health_ports: Dict[str, int]

    consumer_name_prefix: str
    block_ms: int
    batch_size: int

    redis_url: str | None = None  # if set, takes precedence -- supports managed Redis (rediss://user:pass@host:port/db). Optional/defaulted so existing direct RuntimePolicyConfig(...) constructions (tests) remain unaffected.

    @classmethod
    def load(cls, path: str | Path = DEFAULT_POLICY_PATH) -> "RuntimePolicyConfig":
        with open(path) as f:
            raw = json.load(f)
        redis_cfg = raw["redis"]
        # REDIS_URL (managed Redis: rediss://user:pass@host:port/db) takes
        # precedence when set. REDIS_HOST/REDIS_PORT (existing Phase 7
        # env overrides) remain the fallback -- local dev / Docker Compose
        # behavior is completely unchanged when REDIS_URL is unset.
        redis_url = os.environ.get("REDIS_URL") or None
        redis_host = os.environ.get("REDIS_HOST", redis_cfg["host"])
        redis_port = int(os.environ.get("REDIS_PORT", redis_cfg["port"]))
        retry = raw["retry"]
        pending = raw["pending_recovery"]
        idem = raw["idempotency"]
        concurrency = raw["worker_concurrency"]
        return cls(
            redis_host=redis_host, redis_port=redis_port, redis_db=redis_cfg["db"], redis_url=redis_url,
            streams=raw["streams"], consumer_groups=raw["consumer_groups"],
            max_retries=retry["max_retries"], backoff_seconds=retry["backoff_seconds"],
            backoff_multiplier=retry["backoff_multiplier"],
            min_idle_time_ms=pending["min_idle_time_ms"], claim_batch_size=pending["claim_batch_size"],
            idempotency_key_prefix=idem["key_prefix"], idempotency_ttl_seconds=idem["ttl_seconds"],
            health_ports=raw["health"],
            consumer_name_prefix=concurrency["consumer_name_prefix"], block_ms=concurrency["block_ms"],
            batch_size=concurrency["batch_size"],
        )
