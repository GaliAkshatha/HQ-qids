from src.hybrid.decision_policy import DecisionPolicyConfig


def test_loads_from_real_config_file(repo_root):
    policy = DecisionPolicyConfig.load(repo_root / "config" / "hybrid_decision_policy.json")
    assert policy.quantum_override_confidence_threshold == 0.85
    assert policy.classical_high_confidence_threshold == 0.90
