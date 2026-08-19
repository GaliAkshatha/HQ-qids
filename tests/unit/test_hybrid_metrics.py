from src.contracts import DetectionResult, QuantumResult, RoutingDecision
from src.hybrid.decision_policy import DecisionPolicyConfig
from src.hybrid.hybrid_engine import build_hybrid_decision
from src.hybrid.metrics import HybridMetrics
from src.hybrid.risk_engine import assess_risk
from src.hybrid.risk_policy import RiskPolicyConfig


def make_policy():
    return DecisionPolicyConfig(quantum_override_confidence_threshold=0.85, classical_high_confidence_threshold=0.90)


def make_risk_policy():
    return RiskPolicyConfig(
        combination_weights={"threat_evidence": 0.75, "system_uncertainty": 0.25},
        threat_evidence_weights={"classical_attack_probability": 0.4, "quantum_attack_probability": 0.3, "anomaly_score": 0.3},
        system_uncertainty_weights={"model_disagreement": 0.34, "quantum_conflict": 0.33, "fallback": 0.33},
        low_max=0.25, medium_max=0.50, high_max=0.75, confirmed_attack_min_level="HIGH",
    )


def make_detection(sample_id, prediction="normal", confidence=0.9):
    attack_p = confidence if prediction == "attack" else 1 - confidence
    return DetectionResult(
        sample_id=sample_id, classical_prediction=prediction, classical_confidence=confidence,
        class_probabilities={"normal": 1 - attack_p, "attack": attack_p}, anomaly_score=0.1, model_disagreement=0.05,
    )


def make_routing_success(sample_id, quantum_prediction, quantum_confidence):
    qr = QuantumResult(
        sample_id=sample_id, quantum_model="VQC", status="success",
        quantum_prediction=quantum_prediction, quantum_confidence=quantum_confidence,
        class_probabilities={"attack": quantum_confidence, "normal": 1 - quantum_confidence},
    )
    return RoutingDecision(sample_id=sample_id, decision_status="success", should_invoke_quantum=True, quantum_backend="VQC", quantum_result=qr)


def make_routing_not_invoked(sample_id):
    return RoutingDecision(sample_id=sample_id, decision_status="not_invoked", should_invoke_quantum=False)


def make_routing_fallback(sample_id):
    return RoutingDecision(
        sample_id=sample_id, decision_status="fallback", should_invoke_quantum=True,
        quantum_backend="VQC", fallback_used=True, fallback_reason="timeout",
    )


def test_metrics_compute_confirmation_disagreement_and_change_rates():
    policy = make_policy()
    risk_policy = make_risk_policy()
    metrics = HybridMetrics()

    samples = [
        # agreement
        (make_detection("s1", "attack", 0.9), make_routing_success("s1", "attack", 0.9)),
        (make_detection("s2", "normal", 0.9), make_routing_success("s2", "normal", 0.9)),
        # disagreement, not overridden (retained)
        (make_detection("s3", "normal", 0.6), make_routing_success("s3", "attack", 0.6)),
        # disagreement, overridden (decision change)
        (make_detection("s4", "normal", 0.5), make_routing_success("s4", "attack", 0.95)),
        # not invoked -- shouldn't count toward quantum_successes
        (make_detection("s5", "normal", 0.95), make_routing_not_invoked("s5")),
        # fallback -- counts as failure/fallback, not success
        (make_detection("s6", "attack", 0.95), make_routing_fallback("s6")),
    ]

    for dr, rd in samples:
        hd = build_hybrid_decision(dr, rd, policy)
        risk = assess_risk(dr, hd, risk_policy)
        metrics.record(dr.classical_prediction, hd, risk)

    snap = metrics.snapshot()
    assert snap["total_samples"] == 6
    assert snap["quantum_successes"] == 4  # s1-s4 (resolved success, agree or disagree)
    assert snap["agreements"] == 2  # s1, s2
    assert snap["disagreements"] == 2  # s3, s4
    assert snap["decision_changes"] == 1  # only s4 (s3 retained classical)
    assert snap["classical_fallbacks"] == 1  # s6

    assert snap["quantum_confirmation_rate"] == 2 / 4
    assert snap["quantum_disagreement_rate"] == 2 / 4
    assert snap["quantum_decision_change_rate"] == 1 / 4
    # confirmation + disagreement should account for all successful verifications
    assert snap["quantum_confirmation_rate"] + snap["quantum_disagreement_rate"] == 1.0


def test_metrics_rates_are_none_when_no_quantum_successes():
    metrics = HybridMetrics()
    policy = make_policy()
    risk_policy = make_risk_policy()

    dr = make_detection("s1", "normal", 0.95)
    rd = make_routing_not_invoked("s1")
    hd = build_hybrid_decision(dr, rd, policy)
    risk = assess_risk(dr, hd, risk_policy)
    metrics.record(dr.classical_prediction, hd, risk)

    snap = metrics.snapshot()
    assert snap["quantum_successes"] == 0
    assert snap["quantum_confirmation_rate"] is None
    assert snap["quantum_disagreement_rate"] is None
    assert snap["quantum_decision_change_rate"] is None
