import pytest

from src.contracts import DetectionResult
from src.routing.policy import RoutingPolicyConfig, evaluate_routing


def make_policy(**overrides) -> RoutingPolicyConfig:
    base = dict(
        confidence_threshold=0.70,
        anomaly_threshold=0.70,
        disagreement_threshold=0.30,
        combination="any",
        quantum_backend="VQC",
        circuit_breaker_failure_threshold=3,
        circuit_breaker_cooldown_seconds=30,
        timeout_seconds={"QSVM": 5.0, "VQC": 1.0},
        max_retries=2,
        backoff_seconds=0.5,
        queue_max_workers=4,
    )
    base.update(overrides)
    return RoutingPolicyConfig(**base)


def make_detection(confidence=0.95, anomaly=0.1, disagreement=0.05) -> DetectionResult:
    return DetectionResult(
        sample_id="s1",
        classical_prediction="normal",
        classical_confidence=confidence,
        class_probabilities={"normal": confidence, "attack": 1 - confidence},
        anomaly_score=anomaly,
        model_disagreement=disagreement,
    )


def test_confident_low_anomaly_low_disagreement_does_not_invoke():
    policy = make_policy()
    dr = make_detection(confidence=0.95, anomaly=0.1, disagreement=0.05)
    should_invoke, reasons, signals = evaluate_routing(dr, policy)
    assert should_invoke is False
    assert reasons == []
    assert signals["classical_confidence"] == 0.95


def test_low_confidence_alone_triggers_with_correct_reason():
    policy = make_policy()
    dr = make_detection(confidence=0.5, anomaly=0.1, disagreement=0.05)
    should_invoke, reasons, _ = evaluate_routing(dr, policy)
    assert should_invoke is True
    assert reasons == ["LOW_CONFIDENCE"]


def test_high_anomaly_alone_triggers_with_correct_reason():
    policy = make_policy()
    dr = make_detection(confidence=0.95, anomaly=0.9, disagreement=0.05)
    should_invoke, reasons, _ = evaluate_routing(dr, policy)
    assert should_invoke is True
    assert reasons == ["HIGH_ANOMALY"]


def test_high_disagreement_alone_triggers_with_correct_reason():
    policy = make_policy()
    dr = make_detection(confidence=0.95, anomaly=0.1, disagreement=0.5)
    should_invoke, reasons, _ = evaluate_routing(dr, policy)
    assert should_invoke is True
    assert reasons == ["HIGH_DISAGREEMENT"]


def test_multiple_signals_all_recorded():
    policy = make_policy()
    dr = make_detection(confidence=0.5, anomaly=0.9, disagreement=0.5)
    should_invoke, reasons, _ = evaluate_routing(dr, policy)
    assert should_invoke is True
    assert set(reasons) == {"LOW_CONFIDENCE", "HIGH_ANOMALY", "HIGH_DISAGREEMENT"}


def test_boundary_values_are_strict_not_inclusive():
    """confidence exactly at threshold should NOT trigger (< not <=);
    anomaly/disagreement exactly at threshold should NOT trigger (> not >=)."""
    policy = make_policy()
    dr = make_detection(confidence=0.70, anomaly=0.70, disagreement=0.30)
    should_invoke, reasons, _ = evaluate_routing(dr, policy)
    assert should_invoke is False
    assert reasons == []


def test_unsupported_combination_mode_raises():
    policy = make_policy(combination="weighted")
    dr = make_detection()
    with pytest.raises(ValueError):
        evaluate_routing(dr, policy)


def test_policy_loads_from_real_config_file(repo_root):
    policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    assert policy.confidence_threshold == 0.70
    assert policy.anomaly_threshold == 0.70
    assert policy.disagreement_threshold == 0.30
    assert policy.combination == "any"
    assert policy.quantum_backend == "VQC"


def test_with_overrides_returns_modified_copy_without_mutating_original(repo_root):
    policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    qsvm_policy = policy.with_overrides(quantum_backend="QSVM")
    assert qsvm_policy.quantum_backend == "QSVM"
    assert policy.quantum_backend == "VQC"  # original untouched
