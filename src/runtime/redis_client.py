"""
src/runtime/redis_client.py

Thin wrapper around redis-py. Nothing domain-specific lives here --
just connection construction and a connectivity check reusable by every
worker's /ready endpoint and by tests.
"""

from __future__ import annotations

import redis

from src.runtime.config import RuntimePolicyConfig


def build_redis_client(config: RuntimePolicyConfig) -> redis.Redis:
    if config.redis_url:
        return redis.Redis.from_url(config.redis_url, decode_responses=True)
    return redis.Redis(
        host=config.redis_host, port=config.redis_port, db=config.redis_db,
        decode_responses=True,
    )


def check_redis_connectivity(client: redis.Redis) -> bool:
    try:
        return bool(client.ping())
    except redis.exceptions.RedisError:
        return False
