"""
Stage D backend tests. Uses Flask's test client against the real app
factory (create_app()) -- real experiment_service, real domain
components, no mocking of the pipeline itself.
"""

import json

import pytest

from src.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "alive"


def test_ready(client):
    r = client.get("/api/ready")
    assert r.status_code in (200, 503)
    body = r.get_json()
    assert "checks" in body
    assert "redis" in body["checks"]


def test_list_scenarios_matches_real_catalog(client):
    r = client.get("/api/experiments")
    body = r.get_json()
    assert "normal_browsing" in body["scenarios"]
    assert "neptune_flood" in body["scenarios"]


def test_experiment_lifecycle_normal(client):
    r = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 2, "mode": "normal", "quantum": "auto"})
    assert r.status_code == 201
    exp = r.get_json()
    assert exp["status"] == "completed"
    assert exp["incident_count"] == 2

    r2 = client.get(f"/api/experiments/{exp['experiment_id']}")
    assert r2.status_code == 200
    detail = r2.get_json()
    assert len(detail["incidents"]) == 2
    for inc in detail["incidents"]:
        assert inc["current_state"] in ("RESOLVED", "ESCALATED")


def test_experiment_lifecycle_adversarial_with_quantum(client):
    r = client.post("/api/experiments/start", json={"scenario": "neptune_flood", "n_sessions": 1, "mode": "adversarial", "quantum": "auto"})
    assert r.status_code == 201
    exp = r.get_json()
    assert exp["status"] == "completed"


def test_experiment_bounded_session_count(client):
    r = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 999, "mode": "normal", "quantum": "auto"})
    assert r.status_code == 400


def test_experiment_rejects_unknown_scenario(client):
    r = client.post("/api/experiments/start", json={"scenario": "not_a_real_scenario", "n_sessions": 1, "mode": "normal", "quantum": "auto"})
    assert r.status_code == 400


def test_experiment_rejects_invalid_mode(client):
    r = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 1, "mode": "chaos", "quantum": "auto"})
    assert r.status_code == 400


def test_get_unknown_experiment_404(client):
    r = client.get("/api/experiments/not-a-real-id")
    assert r.status_code == 404


def test_incident_retrieval_and_timeline(client):
    start = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 1, "mode": "normal", "quantum": "auto"})
    exp = start.get_json()
    detail = client.get(f"/api/experiments/{exp['experiment_id']}").get_json()
    incident_id = detail["incidents"][0]["incident_id"]

    r = client.get(f"/api/incidents/{incident_id}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["incident"]["incident_id"] == incident_id
    assert len(body["timeline"]) > 0
    assert body["timeline"][0]["event_type"] == "DETECTION_CREATED"
    assert body["timeline"][-1]["event_type"] in ("INCIDENT_RESOLVED", "INCIDENT_ESCALATED")


def test_get_unknown_incident_404(client):
    r = client.get("/api/incidents/not-a-real-incident")
    assert r.status_code == 404


def test_events_for_correlation(client):
    start = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 1, "mode": "normal", "quantum": "auto"})
    exp = start.get_json()
    detail = client.get(f"/api/experiments/{exp['experiment_id']}").get_json()
    correlation_id = detail["incidents"][0]["correlation_id"]

    r = client.get(f"/api/events/{correlation_id}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["correlation_id"] == correlation_id
    assert len(body["events"]) > 0


def test_metrics_endpoint_real_data(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert "system_status" in body
    assert "pipeline_metrics" in body
    assert "model_comparison" in body
    assert body["model_comparison"]["dataset_label"] == "AGENT_GENERATED_LABELED_DATA"


def test_agents_endpoint_reflects_real_experiment_history(client):
    r = client.get("/api/agents")
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body["agents"], list)
    agent_types = {a["agent_type"] for a in body["agents"]}
    assert agent_types <= {"normal", "adversarial"}


def test_no_secrets_in_any_api_response(client):
    endpoints = ["/api/health", "/api/ready", "/api/experiments", "/api/agents", "/api/incidents", "/api/metrics"]
    for ep in endpoints:
        r = client.get(ep)
        body_text = json.dumps(r.get_json())
        for forbidden in ("password", "accessToken", "refreshToken", "\"jwt\"", "Bearer "):
            assert forbidden not in body_text, f"{ep} leaked '{forbidden}'"


def test_experiments_start_stop_endpoint_shape(client):
    start = client.post("/api/experiments/start", json={"scenario": "normal_browsing", "n_sessions": 1, "mode": "normal", "quantum": "auto"})
    exp_id = start.get_json()["experiment_id"]
    r = client.post("/api/experiments/stop", json={"experiment_id": exp_id})
    assert r.status_code == 200


def test_stop_unknown_experiment_404(client):
    r = client.post("/api/experiments/stop", json={"experiment_id": "does-not-exist"})
    assert r.status_code == 404
