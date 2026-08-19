import pytest

from src.contracts import DetectionResult, QuantumResult, RoutingDecision
from src.hybrid.decision_policy import DecisionPolicyConfig
from src.hybrid.hybrid_engine import build_hybrid_decision


def make_policy(**overrides):
    base = dict(quantum_override_confidence_threshold=0.85, classical_high_confidence_threshold=0.90)
    base.update(overrides)
    return DecisionPolicyConfig(**base)


def make_detection(sample_id="s1", prediction="normal", confidence=0.95, anomaly=0.1, disagreement=0.05):
    attack_p = confidence if prediction == "attack" else 1 - confidence
    return DetectionResult(
        sample_id=sample_id, classical_prediction=prediction, classical_confidence=confidence,
        class_probabilities={"normal": 1 - attack_p, "attack": attack_p},
        anomaly_score=anomaly, model_disagreement=disagreement,
    )


def make_routing_not_invoked(sample_id="s1", reason_codes=None):
    return RoutingDecision(
        sample_id=sample_id, decision_status="not_invoked", should_invoke_quantum=False,
        reason_codes=reason_codes or [],
    )


def make_routing_fallback(sample_id="s1", fallback_reason="retries_exhausted", backend="VQC"):
    return RoutingDecision(
        sample_id=sample_id, decision_status="fallback", should_invoke_quantum=True,
        reason_codes=["HIGH_ANOMALY"], quantum_backend=backend,
        fallback_used=True, fallback_reason=fallback_reason,
    )


def make_routing_success(sample_id="s1", quantum_prediction="attack", quantum_confidence=0.9, backend="VQC"):
    qr = QuantumResult(
        sample_id=sample_id, quantum_model=backend, status="success",
        quantum_prediction=quantum_prediction, quantum_confidence=quantum_confidence,
        class_probabilities={
            "attack": quantum_confidence if quantum_prediction == "attack" else 1 - quantum_confidence,
            "normal": 1 - quantum_confidence if quantum_prediction == "attack" else quantum_confidence,
        },
    )
    return RoutingDecision(
        sample_id=sample_id, decision_status="success", should_invoke_quantum=True,
        reason_codes=["HIGH_ANOMALY"], quantum_backend=backend, quantum_attempted=True, quantum_result=qr,
    )


# ---- classical-only cases ------------------------------------------------

def test_classical_only_normal():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.95)
    rd = make_routing_not_invoked()
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "normal"
    assert hd.quantum_used is False
    assert hd.decision_status == "normal"
    assert "CLASSICAL_ONLY_HIGH_CONFIDENCE" in hd.verification_reason
    assert hd.evidence["decision_changed_by_quantum"] is False


def test_classical_only_attack_high_confidence_is_confirmed():
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.95)
    rd = make_routing_not_invoked()
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"
    assert hd.decision_status == "confirmed"


def test_classical_only_attack_low_confidence_is_uncertain():
    """Below classical_high_confidence_threshold -> insufficient evidence,
    even though quantum was never invoked (not_invoked implies routing's
    own, lower, confidence_threshold was cleared -- but hybrid policy's
    bar is higher, 0.90 by default)."""
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.75)  # above routing's 0.70 but below hybrid's 0.90
    rd = make_routing_not_invoked()
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"
    assert hd.decision_status == "uncertain"  # NOT "confirmed" merely because prediction == attack


# ---- quantum confirmation -------------------------------------------------

def test_quantum_confirmation_attack():
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.6)
    rd = make_routing_success(quantum_prediction="attack", quantum_confidence=0.9)
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"
    assert hd.quantum_used is True
    assert hd.decision_status == "confirmed"
    assert "QUANTUM_CONFIRMED_CLASSICAL" in hd.verification_reason
    assert hd.evidence["agreement"] is True
    assert hd.evidence["decision_changed_by_quantum"] is False
    assert hd.final_confidence == max(0.6, 0.9)


def test_quantum_confirmation_normal():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.6)
    rd = make_routing_success(quantum_prediction="normal", quantum_confidence=0.8)
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "normal"
    assert hd.decision_status == "normal"


# ---- quantum override ------------------------------------------------------

def test_quantum_override_when_confidence_clears_threshold():
    policy = make_policy(quantum_override_confidence_threshold=0.85)
    dr = make_detection(prediction="normal", confidence=0.6)
    rd = make_routing_success(quantum_prediction="attack", quantum_confidence=0.90)
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"  # overridden
    assert hd.decision_status == "confirmed"
    assert "QUANTUM_OVERRULED_CLASSICAL" in hd.verification_reason
    assert hd.evidence["decision_changed_by_quantum"] is True
    assert hd.evidence["quantum_override_applied"] is True
    assert hd.final_confidence == 0.90


# ---- quantum disagreement, retained (Case 6) -------------------------------

def test_quantum_disagreement_below_threshold_retains_classical():
    policy = make_policy(quantum_override_confidence_threshold=0.85)
    dr = make_detection(prediction="normal", confidence=0.6)
    rd = make_routing_success(quantum_prediction="attack", quantum_confidence=0.70)  # below 0.85
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "normal"  # retained, not overridden
    assert hd.decision_status == "uncertain"
    assert "QUANTUM_UNCERTAIN_CLASSICAL_RETAINED" in hd.verification_reason
    assert hd.evidence["decision_changed_by_quantum"] is False
    assert hd.evidence["quantum_override_applied"] is False


def test_low_confidence_conflict_still_uncertain_even_if_classical_confidence_was_high():
    """Case 6 nuance: decision_status is 'uncertain' because of the
    unresolved conflict itself, even when the retained classical
    confidence was independently high."""
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.97)  # very confident classical
    rd = make_routing_success(quantum_prediction="normal", quantum_confidence=0.5)  # weak, disagreeing
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"  # retained
    assert hd.decision_status == "uncertain"  # still flagged, despite high classical confidence


# ---- quantum failure / fallback --------------------------------------------

def test_quantum_failure_fallback_uses_classical():
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.95)
    rd = make_routing_fallback(fallback_reason="retries_exhausted")
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "attack"
    assert hd.quantum_used is False
    assert hd.quantum_model == "VQC"  # informational, records which backend was attempted
    assert hd.decision_status == "confirmed"  # classical confidence alone clears the bar
    assert "QUANTUM_FAILED_CLASSICAL_FALLBACK" in hd.verification_reason
    assert hd.evidence["fallback_used"] is True
    assert hd.evidence["fallback_reason"] == "retries_exhausted"
    assert hd.evidence["decision_changed_by_quantum"] is False


def test_circuit_open_fallback_uses_classical():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.6)
    rd = make_routing_fallback(fallback_reason="circuit_open")
    hd = build_hybrid_decision(dr, rd, policy)

    assert hd.final_prediction == "normal"
    assert hd.evidence["fallback_reason"] == "circuit_open"


# ---- decision_changed_by_quantum exact formula -----------------------------

def test_decision_changed_by_quantum_true_only_when_quantum_used_and_prediction_differs():
    policy = make_policy()

    # quantum used, changed -> True
    dr1 = make_detection(prediction="normal", confidence=0.5)
    rd1 = make_routing_success(quantum_prediction="attack", quantum_confidence=0.9)
    hd1 = build_hybrid_decision(dr1, rd1, policy)
    assert hd1.evidence["decision_changed_by_quantum"] is True

    # quantum used, agreed -> False
    dr2 = make_detection(prediction="attack", confidence=0.9)
    rd2 = make_routing_success(quantum_prediction="attack", quantum_confidence=0.9)
    hd2 = build_hybrid_decision(dr2, rd2, policy)
    assert hd2.evidence["decision_changed_by_quantum"] is False

    # quantum not used -> False, regardless of anything else
    dr3 = make_detection(prediction="attack", confidence=0.95)
    rd3 = make_routing_not_invoked()
    hd3 = build_hybrid_decision(dr3, rd3, policy)
    assert hd3.evidence["decision_changed_by_quantum"] is False

    # quantum failed (not "used") -> False even though classical says attack
    dr4 = make_detection(prediction="attack", confidence=0.95)
    rd4 = make_routing_fallback()
    hd4 = build_hybrid_decision(dr4, rd4, policy)
    assert hd4.evidence["decision_changed_by_quantum"] is False


# ---- verification_reason composition ---------------------------------------

def test_verification_reason_preserves_routing_reason_codes():
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.5)
    rd = make_routing_success(quantum_prediction="attack", quantum_confidence=0.9)
    rd.reason_codes = ["HIGH_ANOMALY", "HIGH_DISAGREEMENT"]
    hd = build_hybrid_decision(dr, rd, policy)

    assert "HIGH_ANOMALY" in hd.verification_reason
    assert "HIGH_DISAGREEMENT" in hd.verification_reason
    assert "QUANTUM_CONFIRMED_CLASSICAL" in hd.verification_reason


# ---- invalid inputs ---------------------------------------------------------

def test_pending_routing_decision_raises():
    policy = make_policy()
    dr = make_detection()
    rd = RoutingDecision(sample_id="s1", decision_status="pending", should_invoke_quantum=True, job_id="abc")
    with pytest.raises(ValueError):
        build_hybrid_decision(dr, rd, policy)


def test_mismatched_sample_ids_raises():
    policy = make_policy()
    dr = make_detection(sample_id="s1")
    rd = make_routing_not_invoked(sample_id="s2")
    with pytest.raises(ValueError):
        build_hybrid_decision(dr, rd, policy)


# ---- existing HybridDecision contract validation ---------------------------

def test_every_generated_decision_passes_existing_contract_validation():
    """Regression: the hybrid engine must never produce a HybridDecision
    that would fail its own (unmodified) __post_init__ validation --
    constructing it above already proves this per-case, this test just
    makes the intent explicit as its own case."""
    policy = make_policy()
    cases = [
        (make_detection(prediction="normal", confidence=0.95), make_routing_not_invoked()),
        (make_detection(prediction="attack", confidence=0.95), make_routing_fallback()),
        (make_detection(prediction="attack", confidence=0.5), make_routing_success("s1", "attack", 0.9)),
        (make_detection(prediction="normal", confidence=0.5), make_routing_success("s1", "attack", 0.6)),
    ]
    for dr, rd in cases:
        hd = build_hybrid_decision(dr, rd, policy)
        # to_dict() + re-validate by constructing a fresh instance from the same fields
        from src.contracts import HybridDecision
        HybridDecision(**{k: v for k, v in hd.to_dict().items()})
