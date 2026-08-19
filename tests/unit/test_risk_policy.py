import json

import pytest

from src.hybrid.risk_policy import RiskPolicyConfig


def test_loads_from_real_config_file(repo_root):
    policy = RiskPolicyConfig.load(repo_root / "config" / "risk_policy.json")
    assert policy.combination_weights["threat_evidence"] == 0.75
    assert policy.combination_weights["system_uncertainty"] == 0.25
    assert policy.confirmed_attack_min_level == "HIGH"


def test_weight_sum_validation_rejects_bad_combination_weights(tmp_path):
    bad_config = {
        "combination_weights": {"threat_evidence": 0.5, "system_uncertainty": 0.6},  # sums to 1.1
        "threat_evidence_weights": {"classical_attack_probability": 0.4, "quantum_attack_probability": 0.3, "anomaly_score": 0.3},
        "system_uncertainty_weights": {"model_disagreement": 0.34, "quantum_conflict": 0.33, "fallback": 0.33},
        "thresholds": {"low_max": 0.25, "medium_max": 0.50, "high_max": 0.75},
        "floors": {"confirmed_attack_min_level": "HIGH"},
    }
    path = tmp_path / "bad_risk_policy.json"
    path.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError):
        RiskPolicyConfig.load(path)


def test_weight_sum_validation_rejects_bad_threat_evidence_weights(tmp_path):
    bad_config = {
        "combination_weights": {"threat_evidence": 0.75, "system_uncertainty": 0.25},
        "threat_evidence_weights": {"classical_attack_probability": 0.5, "quantum_attack_probability": 0.5, "anomaly_score": 0.5},
        "system_uncertainty_weights": {"model_disagreement": 0.34, "quantum_conflict": 0.33, "fallback": 0.33},
        "thresholds": {"low_max": 0.25, "medium_max": 0.50, "high_max": 0.75},
        "floors": {"confirmed_attack_min_level": "HIGH"},
    }
    path = tmp_path / "bad_risk_policy2.json"
    path.write_text(json.dumps(bad_config))
    with pytest.raises(ValueError):
        RiskPolicyConfig.load(path)
