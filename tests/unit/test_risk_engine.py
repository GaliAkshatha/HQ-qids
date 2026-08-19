import pytest

from src.contracts import DetectionResult, HybridDecision
from src.hybrid.risk_engine import assess_risk
from src.hybrid.risk_policy import RiskPolicyConfig


def make_policy(**overrides):
    base = dict(
        combination_weights={"threat_evidence": 0.75, "system_uncertainty": 0.25},
        threat_evidence_weights={"classical_attack_probability": 0.4, "quantum_attack_probability": 0.3, "anomaly_score": 0.3},
        system_uncertainty_weights={"model_disagreement": 0.34, "quantum_conflict": 0.33, "fallback": 0.33},
        low_max=0.25, medium_max=0.50, high_max=0.75,
        confirmed_attack_min_level="HIGH",
    )
    base.update(overrides)
    return RiskPolicyConfig(**base)


def make_detection(prediction="normal", confidence=0.95, anomaly=0.1, disagreement=0.05):
    attack_p = confidence if prediction == "attack" else 1 - confidence
    return DetectionResult(
        sample_id="s1", classical_prediction=prediction, classical_confidence=confidence,
        class_probabilities={"normal": 1 - attack_p, "attack": attack_p},
        anomaly_score=anomaly, model_disagreement=disagreement,
    )


def make_hybrid_decision(
    final_prediction="normal", final_confidence=0.95, decision_status="normal",
    quantum_used=False, quantum_prediction=None, quantum_confidence=None,
    evidence=None,
):
    return HybridDecision(
        sample_id="s1", final_prediction=final_prediction, final_confidence=final_confidence,
        quantum_used=quantum_used, quantum_model="VQC" if quantum_used else None,
        quantum_prediction=quantum_prediction, quantum_confidence=quantum_confidence,
        decision_status=decision_status,
        evidence=evidence or {"anomaly_score": 0.1, "model_disagreement": 0.05},
    )


# ---- basic low-risk case ---------------------------------------------------

def test_clean_normal_traffic_is_low_risk():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.98, anomaly=0.05, disagreement=0.02)
    hd = make_hybrid_decision(
        final_prediction="normal", decision_status="normal",
        evidence={"anomaly_score": 0.05, "model_disagreement": 0.02, "agreement": None, "fallback_used": False},
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.risk_level == "LOW"
    assert 0.0 <= risk.risk_score < policy.low_max


# ---- fallback does NOT imply threat -----------------------------------------

def test_fallback_alone_does_not_push_risk_to_high_or_critical():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.6, anomaly=0.1, disagreement=0.05)
    hd = make_hybrid_decision(
        final_prediction="normal", decision_status="normal",
        evidence={"anomaly_score": 0.1, "model_disagreement": 0.05, "agreement": None, "fallback_used": True, "fallback_reason": "circuit_open"},
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.risk_level in ("LOW", "MEDIUM")  # fallback nudges, never dominates
    assert risk.contributing_factors["fallback_indicator"] == 1.0
    # system_uncertainty_score reflects it, but threat_evidence_score doesn't
    assert risk.system_uncertainty_score > 0.0


def test_fallback_with_low_threat_evidence_stays_low_risk():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.95, anomaly=0.05, disagreement=0.02)
    hd = make_hybrid_decision(
        final_prediction="normal", decision_status="normal",
        evidence={"anomaly_score": 0.05, "model_disagreement": 0.02, "agreement": None, "fallback_used": True, "fallback_reason": "timeout"},
    )
    risk = assess_risk(dr, hd, policy)
    # Even with fallback (uncertainty=1.0 contribution to that component),
    # weighted at only 0.25*0.33 ~= 0.08 of total score -- shouldn't reach HIGH
    assert risk.risk_level in ("LOW", "MEDIUM")


# ---- confirmed attack floor --------------------------------------------------

def test_confirmed_attack_gets_at_least_high_floor():
    policy = make_policy()
    # deliberately weak-looking raw signals, but decision_status=confirmed
    dr = make_detection(prediction="attack", confidence=0.91, anomaly=0.3, disagreement=0.1)
    hd = make_hybrid_decision(
        final_prediction="attack", final_confidence=0.91, decision_status="confirmed",
        evidence={"anomaly_score": 0.3, "model_disagreement": 0.1, "agreement": None, "fallback_used": False},
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.risk_level in ("HIGH", "CRITICAL")
    if risk.floor_applied:
        assert risk.floor_applied == "confirmed_attack_min_level"


def test_confirmed_attack_is_not_automatically_critical():
    """Point 7: CRITICAL requires substantially stronger evidence than
    just being 'confirmed' -- the floor stops at HIGH, not CRITICAL."""
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.91, anomaly=0.2, disagreement=0.05)
    hd = make_hybrid_decision(
        final_prediction="attack", final_confidence=0.91, decision_status="confirmed",
        evidence={"anomaly_score": 0.2, "model_disagreement": 0.05, "agreement": True, "fallback_used": False},
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.risk_level == "HIGH"  # floored to HIGH, not bumped all the way to CRITICAL
    assert risk.floor_applied == "confirmed_attack_min_level"


def test_strong_evidence_reaches_critical_without_needing_the_floor():
    policy = make_policy()
    dr = make_detection(prediction="attack", confidence=0.97, anomaly=0.95, disagreement=0.4)
    hd = make_hybrid_decision(
        final_prediction="attack", final_confidence=0.97, decision_status="confirmed",
        quantum_used=True, quantum_prediction="attack", quantum_confidence=0.95,
        evidence={"anomaly_score": 0.95, "model_disagreement": 0.4, "agreement": True, "fallback_used": False},
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.risk_level == "CRITICAL"
    assert risk.risk_score >= policy.high_max
    # floor may or may not have been the deciding factor -- score alone should clear it
    assert risk.threat_evidence_score > 0.7


# ---- boundary conditions ------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "LOW"), (0.24, "LOW"), (0.25, "MEDIUM"), (0.49, "MEDIUM"), (0.50, "HIGH"), (0.74, "HIGH"), (0.75, "CRITICAL"), (1.0, "CRITICAL")],
)
def test_risk_level_boundaries_are_strict_lower_inclusive(score, expected):
    from src.hybrid.risk_engine import _level_from_score
    policy = make_policy()
    assert _level_from_score(score, policy) == expected


# ---- quantum conflict indicator ------------------------------------------------

def test_unresolved_quantum_conflict_raises_uncertainty_not_threat():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.6, anomaly=0.1, disagreement=0.05)
    hd = make_hybrid_decision(
        final_prediction="normal", decision_status="uncertain",
        quantum_used=True, quantum_prediction="attack", quantum_confidence=0.6,
        evidence={
            "anomaly_score": 0.1, "model_disagreement": 0.05, "agreement": False,
            "quantum_override_applied": False, "fallback_used": False,
        },
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.contributing_factors["quantum_conflict_indicator"] == 1.0
    assert "UNRESOLVED_QUANTUM_CONFLICT" in risk.reason_codes


def test_resolved_override_does_not_count_as_conflict():
    policy = make_policy()
    dr = make_detection(prediction="normal", confidence=0.6)
    hd = make_hybrid_decision(
        final_prediction="attack", decision_status="confirmed",
        quantum_used=True, quantum_prediction="attack", quantum_confidence=0.9,
        evidence={
            "anomaly_score": 0.1, "model_disagreement": 0.05, "agreement": False,
            "quantum_override_applied": True, "fallback_used": False,
        },
    )
    risk = assess_risk(dr, hd, policy)
    assert risk.contributing_factors["quantum_conflict_indicator"] == 0.0
