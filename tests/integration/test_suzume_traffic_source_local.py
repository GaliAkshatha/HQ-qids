"""
Real HTTP integration tests against tests/support/local_suzume_target.py
-- a real Flask server, real HTTP requests via `requests`, real JWT
issuance/verification, real httpOnly-cookie-based refresh flow. Labeled
CONTROLLED_LOCAL_SUZUME throughout, never REAL_SUZUME_INTERACTION.
"""

import threading
import time
import uuid
from datetime import datetime, timezone

import pytest
import requests

from src.agents.agent_action import ApplicationAgentAction
from src.agents.agent_session import AgentSession
from src.agents.suzume_traffic_source import SuzumeTrafficSource
from tests.support.local_suzume_target import build_local_suzume_app


@pytest.fixture(scope="module")
def local_target_url():
    app = build_local_suzume_app()
    port = 5199

    def run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1.0)
    yield f"http://127.0.0.1:{port}"


def make_session(agent_type="normal"):
    return AgentSession(
        session_id=str(uuid.uuid4()), agent_id="test-agent", agent_type=agent_type,
        target_label="CONTROLLED_LOCAL_SUZUME", created_at=datetime.now(timezone.utc).isoformat(),
    )


def unique_email(prefix="agent"):
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


def test_register_authenticates_the_session(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    obs = source.execute_action(session, ApplicationAgentAction(
        action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"},
    ))
    assert obs.status_code == 201
    assert obs.target_label == "CONTROLLED_LOCAL_SUZUME"
    assert session.is_authenticated()
    assert session.access_token is not None


def test_login_with_wrong_password_fails_and_does_not_authenticate(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    email = unique_email()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": email, "password": "password123"}))
    session.clear_authentication()

    obs = source.execute_action(session, ApplicationAgentAction(action_type="login", payload={"email": email, "password": "WrongPassword"}))
    assert obs.status_code == 401
    assert session.is_authenticated() is False


def test_login_with_correct_password_succeeds(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    email = unique_email()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": email, "password": "password123"}))
    session.clear_authentication()

    obs = source.execute_action(session, ApplicationAgentAction(action_type="login", payload={"email": email, "password": "password123"}))
    assert obs.status_code == 200
    assert session.is_authenticated()


def test_refresh_reissues_a_valid_access_token(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))

    obs = source.execute_action(session, ApplicationAgentAction(action_type="refresh"))
    assert obs.status_code == 200
    assert session.is_authenticated()
    assert session.access_token is not None


def test_automatic_refresh_and_retry_on_401_for_non_auth_endpoint(local_target_url):
    """The critical auth-flow behavior: a stale/invalid access token on a
    protected non-auth endpoint triggers exactly one silent refresh, then
    a successful retry -- mirroring the real frontend's client.ts logic."""
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))

    session.access_token = "deliberately-invalid-token"
    obs = source.execute_action(session, ApplicationAgentAction(action_type="list_applications"))

    assert obs.status_code == 200  # succeeded after transparent refresh+retry
    assert session.access_token != "deliberately-invalid-token"


def test_logout_clears_local_authentication_state(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))
    assert session.is_authenticated()

    obs = source.execute_action(session, ApplicationAgentAction(action_type="logout"))
    assert obs.status_code == 204
    assert session.is_authenticated() is False


def test_malformed_registration_payload_produces_validation_failure(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    obs = source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": "not-an-email", "password": "short"}))
    assert obs.status_code == 400
    assert obs.validation_success is False


def test_full_real_workflow_create_application_round_experience_question(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))

    app_obs = source.execute_action(session, ApplicationAgentAction(action_type="create_application", payload={"companyName": "Acme", "role": "SDE"}))
    assert app_obs.status_code == 201
    application_id = session.last_response_data["application"]["id"]

    round_obs = source.execute_action(session, ApplicationAgentAction(
        action_type="create_round", parent_id=application_id, payload={"title": "OA", "type": "ONLINE_ASSESSMENT"},
    ))
    assert round_obs.status_code == 201
    round_id = session.last_response_data["round"]["id"]

    exp_obs = source.execute_action(session, ApplicationAgentAction(
        action_type="create_experience", parent_id=round_id, payload={"summary": "went ok", "confidence": 7},
    ))
    assert exp_obs.status_code == 201
    experience_id = session.last_response_data["experience"]["id"]

    q_obs = source.execute_action(session, ApplicationAgentAction(
        action_type="create_question", parent_id=experience_id, payload={"question": "explain a hash map", "category": "DSA"},
    ))
    assert q_obs.status_code == 201


def test_get_application_with_nonexistent_id_returns_404(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))

    obs = source.execute_action(session, ApplicationAgentAction(action_type="get_application", target_id=str(uuid.uuid4())))
    assert obs.status_code == 404


def test_response_never_contains_raw_secrets_in_observation(local_target_url):
    """Structural proof: even though the real response contains an
    accessToken, the returned ApplicationObservation never carries it."""
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    obs = source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))
    obs_dict = obs.to_dict()
    serialized = str(obs_dict)
    assert session.access_token not in serialized


def test_last_response_data_has_token_redacted(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    session = make_session()
    source.execute_action(session, ApplicationAgentAction(action_type="register", payload={"name": "Test", "email": unique_email(), "password": "password123"}))
    assert "accessToken" not in session.last_response_data
    assert "user" in session.last_response_data  # non-secret data preserved
