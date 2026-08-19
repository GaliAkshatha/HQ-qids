"""
Tests for Stage E (deployment readiness) config behaviors: CORS
allow-list, REDIS_URL precedence over host/port, event store backend
selection, and the dashboard_service Redis-check bug fix.
"""

from src.api.app import create_app
from src.api.services import dashboard_service
from src.runtime.config import RuntimePolicyConfig


def test_cors_defaults_to_wildcard_when_unset(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    app = create_app()
    client = app.test_client()
    r = client.get("/api/health")
    assert r.headers["Access-Control-Allow-Origin"] == "*"


def test_cors_allows_configured_origin(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://qids.vercel.app,https://qids-staging.vercel.app")
    app = create_app()
    client = app.test_client()
    r = client.get("/api/health", headers={"Origin": "https://qids.vercel.app"})
    assert r.headers["Access-Control-Allow-Origin"] == "https://qids.vercel.app"


def test_cors_rejects_unconfigured_origin(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://qids.vercel.app")
    app = create_app()
    client = app.test_client()
    r = client.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert "Access-Control-Allow-Origin" not in r.headers


def test_redis_url_takes_precedence_over_host_port(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://user:pass@managed-redis-host:6380/2")
    monkeypatch.setenv("REDIS_HOST", "should-be-ignored")
    monkeypatch.setenv("REDIS_PORT", "9999")
    config = RuntimePolicyConfig.load()
    assert config.redis_url == "redis://user:pass@managed-redis-host:6380/2"


def test_redis_url_unset_falls_back_to_host_port(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "some-host")
    monkeypatch.setenv("REDIS_PORT", "1234")
    config = RuntimePolicyConfig.load()
    assert config.redis_url is None
    assert config.redis_host == "some-host"
    assert config.redis_port == 1234


def test_backward_compatible_direct_construction_without_redis_url():
    config = RuntimePolicyConfig(
        redis_host="fake", redis_port=0, redis_db=0,
        streams={}, consumer_groups={},
        max_retries=1, backoff_seconds=0.1, backoff_multiplier=1.0,
        min_idle_time_ms=1, claim_batch_size=1,
        idempotency_key_prefix="x", idempotency_ttl_seconds=1,
        health_ports={}, consumer_name_prefix="x", block_ms=1, batch_size=1,
    )
    assert config.redis_url is None


def test_check_redis_uses_configured_connection_not_bare_default(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "definitely-not-a-real-host-xyz")
    monkeypatch.setenv("REDIS_PORT", "6379")
    assert dashboard_service.check_redis() is False


def test_event_store_backend_defaults_to_jsonl(monkeypatch, tmp_path):
    monkeypatch.delenv("EVENT_STORE_BACKEND", raising=False)
    monkeypatch.setenv("EVENT_STORE_PATH", str(tmp_path / "events.jsonl"))
    from src.api.services.experiment_service import _build_event_store
    from src.incident.event_store import JsonlEventStore

    store = _build_event_store()
    assert isinstance(store, JsonlEventStore)
    assert store.path == tmp_path / "events.jsonl"


def test_event_store_path_is_configurable(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_dir" / "custom_events.jsonl"
    monkeypatch.setenv("EVENT_STORE_PATH", str(custom_path))
    monkeypatch.delenv("EVENT_STORE_BACKEND", raising=False)
    from src.api.services.experiment_service import _build_event_store

    store = _build_event_store()
    assert store.path == custom_path


def test_event_store_backend_redis_selects_redis_event_store(monkeypatch):
    monkeypatch.setenv("EVENT_STORE_BACKEND", "redis")
    from src.api.services.experiment_service import _build_event_store
    from src.runtime.event_store_redis import RedisEventStore

    store = _build_event_store()
    assert isinstance(store, RedisEventStore)


def test_port_env_var_is_read_in_app_module():
    import src.api.app as app_module

    source = open(app_module.__file__).read()
    assert 'os.environ.get("PORT"' in source
