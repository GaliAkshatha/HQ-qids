import json

import pytest

from src.agents.scenario_catalog import EnvironmentPolicy, ScenarioCatalog, ScenarioCatalogError


def test_loads_real_scenario_catalog(repo_root):
    catalog = ScenarioCatalog.load(repo_root / "config" / "agent_scenarios.json")
    assert "normal_browsing" in catalog.names()
    assert "neptune_flood" in catalog.names()
    normal = catalog.get("normal_browsing")
    assert normal.category == "normal"
    assert normal.source_label == "normal"


def test_get_unknown_scenario_raises(repo_root):
    catalog = ScenarioCatalog.load(repo_root / "config" / "agent_scenarios.json")
    with pytest.raises(ScenarioCatalogError):
        catalog.get("does_not_exist")


def test_malformed_scenario_config_missing_scenarios_key_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_scenarios": {}}))
    with pytest.raises(ScenarioCatalogError):
        ScenarioCatalog.load(path)


def test_malformed_scenario_config_empty_scenarios_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"scenarios": {}}))
    with pytest.raises(ScenarioCatalogError):
        ScenarioCatalog.load(path)


def test_malformed_scenario_missing_required_field_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"scenarios": {"x": {"category": "attack", "source_label": "neptune"}}}))
    with pytest.raises(ScenarioCatalogError):
        ScenarioCatalog.load(path)


def test_malformed_scenario_invalid_category_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "scenarios": {"x": {"category": "not_valid", "source_label": "neptune", "description": "d", "default_perturbation": 0.1}}
    }))
    with pytest.raises(ScenarioCatalogError):
        ScenarioCatalog.load(path)


def test_loads_real_environment_policy(repo_root):
    policy = EnvironmentPolicy.load(repo_root / "config" / "agent_environment_policy.json")
    assert policy.default_turns > 0
    assert "normal_browsing" in policy.normal_allowed_scenarios
    assert "neptune_flood" in policy.adversarial_allowed_scenarios
    assert 0.0 <= policy.perturbation_default <= policy.perturbation_max


def test_malformed_environment_policy_missing_section_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"run": {"default_turns": 5, "normal_agent_weight": 0.5, "adversarial_agent_weight": 0.5}}))
    with pytest.raises(ScenarioCatalogError):
        EnvironmentPolicy.load(path)
