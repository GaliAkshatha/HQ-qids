import redis

from src.runtime.config import RuntimePolicyConfig
from src.runtime.redis_client import build_redis_client, check_redis_connectivity


def test_real_redis_connection_and_ping():
    config = RuntimePolicyConfig.load()
    client = build_redis_client(config)
    assert check_redis_connectivity(client) is True
    assert isinstance(client, redis.Redis)


def test_check_connectivity_returns_false_for_unreachable_redis():
    bad_client = redis.Redis(host="localhost", port=59999, socket_connect_timeout=0.2)
    assert check_redis_connectivity(bad_client) is False
