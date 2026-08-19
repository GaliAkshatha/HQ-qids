import pytest

from src.defense.defense_policy import DefensePolicyConfig


def test_loads_from_real_config_file(repo_root):
    policy = DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json")
    assert policy.simulation_mode is True
    assert policy.uncertain_requires_reversible_action is True
    assert policy.max_recovery_retries == 1


def test_low_risk_maps_to_monitor_regardless_of_decision_status(repo_root):
    policy = DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json")
    for status in ("normal", "confirmed", "uncertain"):
        assert policy.select_action("LOW", status) == "MONITOR"


def test_high_risk_confirmed_gets_stronger_action_than_uncertain(repo_root):
    """The approved tiebreaker: 'confirmed' (solid evidence) gets the
    stronger option; 'uncertain' gets the milder, reversible one."""
    policy = DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json")
    confirmed_action = policy.select_action("HIGH", "confirmed")
    uncertain_action = policy.select_action("HIGH", "uncertain")
    assert confirmed_action == "ISOLATE_SIMULATED_SOURCE"
    assert uncertain_action == "RATE_LIMIT"
    assert confirmed_action != uncertain_action


def test_critical_confirmed_gets_block_uncertain_gets_isolate(repo_root):
    policy = DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json")
    assert policy.select_action("CRITICAL", "confirmed") == "BLOCK_SIMULATED_SOURCE"
    assert policy.select_action("CRITICAL", "uncertain") == "ISOLATE_SIMULATED_SOURCE"


def test_unmapped_risk_level_raises(repo_root):
    policy = DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json")
    with pytest.raises(ValueError):
        policy.select_action("NOT_A_LEVEL", "confirmed")


def test_config_missing_risk_level_entry_raises(tmp_path):
    import json

    bad = {
        "risk_action_map": {
            "LOW": {"normal": "MONITOR", "confirmed": "MONITOR", "uncertain": "MONITOR"},
            # missing MEDIUM, HIGH, CRITICAL
        },
        "disabled_actions": [], "uncertain_requires_reversible_action": True,
        "recovery": {"max_retries": 1, "rollback_on_failure": True}, "simulation_mode": True,
    }
    path = tmp_path / "bad_defense_policy.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        DefensePolicyConfig.load(path)
