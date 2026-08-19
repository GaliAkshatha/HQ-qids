import random

import pytest

from src.agents.adversarial_agent import AdversarialAgent, ScenarioNotAllowedError
from src.agents.normal_agent import NormalUserAgent


def test_normal_agent_only_selects_from_allowed_scenarios():
    agent = NormalUserAgent("n1", ("normal_browsing",))
    rng = random.Random(0)
    for _ in range(20):
        action = agent.act("sess1", rng)
        assert action.scenario_name == "normal_browsing"
        assert action.agent_type == "normal"


def test_adversarial_agent_only_selects_from_allow_list():
    allowed = ("neptune_flood", "satan_scan")
    agent = AdversarialAgent("a1", allowed)
    rng = random.Random(1)
    for _ in range(30):
        action = agent.act("sess1", rng)
        assert action.scenario_name in allowed


def test_adversarial_agent_rejects_empty_allow_list():
    with pytest.raises(ScenarioNotAllowedError):
        AdversarialAgent("a1", ())


def test_adversarial_agent_request_scenario_enforces_allow_list():
    agent = AdversarialAgent("a1", ("neptune_flood",))
    action = agent.request_scenario("neptune_flood", "sess1")
    assert action.scenario_name == "neptune_flood"

    with pytest.raises(ScenarioNotAllowedError):
        agent.request_scenario("smurf_flood", "sess1")


def test_adversarial_agent_cannot_be_tricked_into_disallowed_scenario_via_request():
    agent = AdversarialAgent("a1", ("satan_scan",))
    with pytest.raises(ScenarioNotAllowedError):
        agent.request_scenario("BLOCK_SIMULATED_SOURCE", "sess1")
    with pytest.raises(ScenarioNotAllowedError):
        agent.request_scenario("neptune_flood", "sess1")


def test_no_agent_module_has_network_socket_subprocess_capability():
    import src.agents.agent_base as base_mod
    import src.agents.normal_agent as normal_mod
    import src.agents.adversarial_agent as adversarial_mod
    from tests.unit._agent_safety_check import assert_no_io_capability

    for mod in (base_mod, normal_mod, adversarial_mod):
        assert_no_io_capability(mod.__file__)
