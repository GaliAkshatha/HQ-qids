"""
Real integration tests for SuzumeAdversarialAgent and SuzumeNormalAgent
against the local controlled target -- proving bounded behavior, correct
scenario enforcement, deterministic seeding, and that feature extraction
on the resulting real telemetry produces the expected security signal.
"""

import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

from src.agents.adversarial_agent import (
    SUZUME_ADVERSARIAL_SCENARIOS,
    ScenarioNotAllowedError,
    SuzumeAdversarialAgent,
)
from src.agents.agent_session import AgentSession
from src.agents.application_features import compute_application_features
from src.agents.normal_agent import SuzumeNormalAgent
from src.agents.suzume_traffic_source import SuzumeTrafficSource
from tests.support.local_suzume_target import build_local_suzume_app


@pytest.fixture(scope="module")
def local_target_url():
    app = build_local_suzume_app()
    port = 5198

    def run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1.0)
    yield f"http://127.0.0.1:{port}"


def make_session(agent_id="test-agent", agent_type="adversarial"):
    return AgentSession(
        session_id=str(uuid.uuid4()), agent_id=agent_id, agent_type=agent_type,
        target_label="CONTROLLED_LOCAL_SUZUME", created_at=datetime.now(timezone.utc).isoformat(),
    )


def test_normal_agent_full_workflow_produces_only_successful_real_observations(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeNormalAgent("normal-1", seed=1)
    session = make_session(agent_type="normal")

    observations = agent.run_session(source, session)

    assert len(observations) == 9
    for obs in observations:
        assert obs.status_code is not None and obs.status_code < 400
        assert obs.target_label == "CONTROLLED_LOCAL_SUZUME"
        assert obs.agent_type == "normal"


def test_normal_agent_is_deterministic_with_fixed_seed(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")

    agent1 = SuzumeNormalAgent("normal-a", seed=99)
    session1 = make_session(agent_type="normal")
    obs1 = agent1.run_session(source, session1)

    agent2 = SuzumeNormalAgent("normal-b", seed=99)
    session2 = make_session(agent_type="normal")
    obs2 = agent2.run_session(source, session2)

    action_types_1 = [o.action_type for o in obs1]
    action_types_2 = [o.action_type for o in obs2]
    assert action_types_1 == action_types_2


def test_normal_agent_features_show_low_anomaly_signal(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeNormalAgent("normal-2", seed=5)
    session = make_session(agent_type="normal")
    observations = agent.run_session(source, session)

    vec = compute_application_features(observations)
    assert vec.failed_auth_rate == 0.0
    assert vec.validation_failure_rate == 0.0
    assert vec.invalid_resource_rate == 0.0
    assert vec.response_error_rate == 0.0


def test_adversarial_agent_rejects_disallowed_scenario(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeAdversarialAgent("adv-1", allowed_scenario_ids=("malformed_payload_probe",), seed=1)
    session = make_session()
    with pytest.raises(ScenarioNotAllowedError):
        agent.run_session(source, session, scenario_id="repeated_failed_login")


def test_adversarial_agent_rejects_unknown_scenario_at_construction():
    with pytest.raises(ScenarioNotAllowedError):
        SuzumeAdversarialAgent("adv-2", allowed_scenario_ids=("not_a_real_scenario",))


def test_adversarial_agent_rejects_empty_allow_list():
    with pytest.raises(ScenarioNotAllowedError):
        SuzumeAdversarialAgent("adv-3", allowed_scenario_ids=())


def test_repeated_failed_login_scenario_stays_within_bound_and_matches_pattern(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeAdversarialAgent("adv-4", allowed_scenario_ids=("repeated_failed_login",), seed=2)
    session = make_session()

    observations = agent.run_session(source, session, scenario_id="repeated_failed_login")

    scenario = SUZUME_ADVERSARIAL_SCENARIOS["repeated_failed_login"]
    login_obs = [o for o in observations if o.action_type == "login"]
    assert len(login_obs) <= scenario.max_actions
    assert all(o.status_code == 401 for o in login_obs)

    vec = compute_application_features(observations)
    assert vec.failed_auth_rate > 0.5
    assert vec.auth_failure_burst > 0.5


def test_malformed_payload_probe_scenario_all_rejected_by_real_validation(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeAdversarialAgent("adv-5", allowed_scenario_ids=("malformed_payload_probe",), seed=3)
    session = make_session()

    observations = agent.run_session(source, session, scenario_id="malformed_payload_probe")
    scenario = SUZUME_ADVERSARIAL_SCENARIOS["malformed_payload_probe"]

    create_obs = [o for o in observations if o.action_type == "create_application"]
    assert len(create_obs) <= scenario.max_actions
    assert all(o.status_code == 400 for o in create_obs)
    assert all(not o.validation_success for o in create_obs)

    vec = compute_application_features(observations)
    assert vec.validation_failure_rate > 0.5


def test_invalid_resource_probe_scenario_all_return_404(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeAdversarialAgent("adv-6", allowed_scenario_ids=("invalid_resource_probe",), seed=4)
    session = make_session()

    observations = agent.run_session(source, session, scenario_id="invalid_resource_probe")
    scenario = SUZUME_ADVERSARIAL_SCENARIOS["invalid_resource_probe"]

    get_obs = [o for o in observations if o.action_type == "get_application"]
    assert len(get_obs) <= scenario.max_actions
    assert all(o.status_code == 404 for o in get_obs)

    vec = compute_application_features(observations)
    assert vec.invalid_resource_rate > 0.5


def test_rapid_endpoint_switching_scenario_is_read_only_and_bounded(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    agent = SuzumeAdversarialAgent("adv-7", allowed_scenario_ids=("rapid_endpoint_switching",), seed=6)
    session = make_session()

    observations = agent.run_session(source, session, scenario_id="rapid_endpoint_switching")
    scenario = SUZUME_ADVERSARIAL_SCENARIOS["rapid_endpoint_switching"]

    non_register_obs = [o for o in observations if o.action_type != "register"]
    assert len(non_register_obs) <= scenario.max_actions
    assert all(not o.action_type.startswith(("create_", "update_", "delete_")) for o in non_register_obs)
    assert all(o.method == "GET" for o in non_register_obs)

    vec = compute_application_features(observations)
    assert vec.endpoint_switch_rate > 0.5


def test_every_scenario_has_full_required_documentation():
    for scenario in SUZUME_ADVERSARIAL_SCENARIOS.values():
        assert scenario.scenario_id
        assert scenario.description
        assert scenario.max_actions > 0
        assert scenario.reason
        assert scenario.expected_telemetry_pattern


def test_adversarial_agent_never_performs_a_real_destructive_action(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    for scenario_id in SUZUME_ADVERSARIAL_SCENARIOS:
        agent = SuzumeAdversarialAgent(f"adv-safety-{scenario_id}", allowed_scenario_ids=(scenario_id,), seed=0)
        session = make_session()
        observations = agent.run_session(source, session, scenario_id=scenario_id)
        for obs in observations:
            assert not obs.action_type.startswith(("update_", "delete_")), (
                f"scenario '{scenario_id}' issued a destructive action: {obs.action_type}"
            )
