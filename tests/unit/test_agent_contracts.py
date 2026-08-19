import pytest

from src.agents.contracts import AgentProfile, GeneratedTrafficRecord, ScenarioDefinition


def test_scenario_definition_valid():
    s = ScenarioDefinition(name="x", category="attack", source_label="neptune", description="d", default_perturbation=0.1)
    assert s.category == "attack"


def test_scenario_definition_rejects_bad_category():
    with pytest.raises(ValueError):
        ScenarioDefinition(name="x", category="malicious", source_label="neptune", description="d", default_perturbation=0.1)


def test_scenario_definition_rejects_out_of_range_perturbation():
    with pytest.raises(ValueError):
        ScenarioDefinition(name="x", category="attack", source_label="neptune", description="d", default_perturbation=1.5)


def test_agent_profile_valid():
    p = AgentProfile(agent_id="a1", agent_type="adversarial", allowed_scenarios=("neptune_flood",))
    assert p.agent_type == "adversarial"


def test_agent_profile_rejects_bad_type():
    with pytest.raises(ValueError):
        AgentProfile(agent_id="a1", agent_type="hacker", allowed_scenarios=("x",))


def test_agent_profile_rejects_empty_allowed_scenarios():
    with pytest.raises(ValueError):
        AgentProfile(agent_id="a1", agent_type="normal", allowed_scenarios=())


def test_generated_traffic_record_to_dict():
    record = GeneratedTrafficRecord(
        sample_id="s1", session_id="sess1", agent_id="a1", agent_type="normal",
        scenario_name="normal_browsing", intended_label="normal", source_exemplar_index=5,
        source_exemplar_label="normal", perturbation_magnitude=0.05, perturbed_fields={},
    )
    d = record.to_dict()
    assert d["sample_id"] == "s1"
    assert d["intended_label"] == "normal"


def test_no_network_socket_or_subprocess_capability_on_contracts_module():
    """Structural safety check via AST (not substring search, which would
    false-positive on this module's own safety-documentation docstrings)."""
    import src.agents.contracts as mod
    from tests.unit._agent_safety_check import assert_no_io_capability
    assert_no_io_capability(mod.__file__)
