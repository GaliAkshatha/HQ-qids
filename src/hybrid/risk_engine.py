"""
src/hybrid/risk_engine.py

Deterministic, config-driven risk scoring. No ML, no LLM, no randomness.

Approved architectural rule: THREAT EVIDENCE and SYSTEM UNCERTAINTY are
kept as two separately-weighted components, combined at the end --
fallback/disagreement/model-disagreement live only in the uncertainty
component (weighted modestly, 0.25 by default) and can never by
themselves push risk into HIGH/CRITICAL, because that requires the
threat_evidence component (weighted 0.75 by default) to independently
carry the score there. A quantum fallback is evidence the system is less
sure, not evidence of an attack.

Floor rule: a "confirmed" attack decision gets a risk_level floor of at
least HIGH (approved). This does NOT mean automatic CRITICAL -- CRITICAL
still requires the raw weighted score to cross high_max on its own merits
(approved point 7).
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts import DetectionResult, HybridDecision, RiskAssessment
from src.hybrid.risk_policy import RiskPolicyConfig

_LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _level_from_score(score: float, policy: RiskPolicyConfig) -> str:
    if score < policy.low_max:
        return "LOW"
    if score < policy.medium_max:
        return "MEDIUM"
    if score < policy.high_max:
        return "HIGH"
    return "CRITICAL"


def _max_level(a: str, b: str) -> str:
    return a if _LEVEL_ORDER.index(a) >= _LEVEL_ORDER.index(b) else b


def assess_risk(
    detection_result: DetectionResult,
    hybrid_decision: HybridDecision,
    policy: RiskPolicyConfig,
) -> RiskAssessment:
    ev = hybrid_decision.evidence

    # ---- threat evidence component --------------------------------------
    classical_attack_probability = detection_result.class_probabilities.get("attack", 0.0)
    quantum_attack_probability = 0.0
    if hybrid_decision.quantum_used and hybrid_decision.quantum_prediction is not None:
        # QuantumResult.class_probabilities isn't threaded onto
        # HybridDecision directly, but quantum_prediction + quantum_confidence
        # are -- reconstruct an attack-probability proxy from them, exact
        # when quantum_prediction=="attack" (confidence IS P(attack)),
        # and (1 - confidence) when it predicted normal.
        quantum_attack_probability = (
            hybrid_decision.quantum_confidence
            if hybrid_decision.quantum_prediction == "attack"
            else 1.0 - hybrid_decision.quantum_confidence
        )

    tw = policy.threat_evidence_weights
    threat_evidence_score = (
        tw["classical_attack_probability"] * classical_attack_probability
        + tw["quantum_attack_probability"] * quantum_attack_probability
        + tw["anomaly_score"] * ev["anomaly_score"]
    )
    threat_evidence_score = min(max(threat_evidence_score, 0.0), 1.0)

    # ---- system uncertainty component ------------------------------------
    quantum_conflict_indicator = 1.0 if ev.get("agreement") is False and not ev.get("quantum_override_applied", False) else 0.0
    fallback_indicator = 1.0 if ev.get("fallback_used") else 0.0

    uw = policy.system_uncertainty_weights
    system_uncertainty_score = (
        uw["model_disagreement"] * ev["model_disagreement"]
        + uw["quantum_conflict"] * quantum_conflict_indicator
        + uw["fallback"] * fallback_indicator
    )
    system_uncertainty_score = min(max(system_uncertainty_score, 0.0), 1.0)

    # ---- combine -----------------------------------------------------------
    cw = policy.combination_weights
    risk_score = cw["threat_evidence"] * threat_evidence_score + cw["system_uncertainty"] * system_uncertainty_score
    risk_score = min(max(risk_score, 0.0), 1.0)

    computed_level = _level_from_score(risk_score, policy)

    floor_applied = None
    risk_level = computed_level
    is_confirmed_attack = hybrid_decision.decision_status == "confirmed" and hybrid_decision.final_prediction == "attack"
    if is_confirmed_attack:
        floored = _max_level(computed_level, policy.confirmed_attack_min_level)
        if floored != computed_level:
            floor_applied = "confirmed_attack_min_level"
        risk_level = floored

    reason_codes = []
    if is_confirmed_attack:
        reason_codes.append("CONFIRMED_ATTACK")
    if quantum_conflict_indicator:
        reason_codes.append("UNRESOLVED_QUANTUM_CONFLICT")
    if fallback_indicator:
        reason_codes.append("QUANTUM_FALLBACK_UNCERTAINTY")
    if ev["anomaly_score"] >= 0.7:
        reason_codes.append("HIGH_ANOMALY_SCORE")
    if ev["model_disagreement"] >= 0.3:
        reason_codes.append("HIGH_MODEL_DISAGREEMENT")
    if floor_applied:
        reason_codes.append("RISK_FLOOR_APPLIED")
    if not reason_codes:
        reason_codes.append("NO_SIGNIFICANT_RISK_FACTORS")

    return RiskAssessment(
        sample_id=hybrid_decision.sample_id,
        risk_level=risk_level,
        risk_score=risk_score,
        threat_evidence_score=threat_evidence_score,
        system_uncertainty_score=system_uncertainty_score,
        reason_codes=reason_codes,
        contributing_factors={
            "classical_attack_probability": classical_attack_probability,
            "quantum_attack_probability": quantum_attack_probability,
            "anomaly_score": ev["anomaly_score"],
            "model_disagreement": ev["model_disagreement"],
            "quantum_conflict_indicator": quantum_conflict_indicator,
            "fallback_indicator": fallback_indicator,
        },
        policy_thresholds={
            "low_max": policy.low_max,
            "medium_max": policy.medium_max,
            "high_max": policy.high_max,
        },
        floor_applied=floor_applied,
        timestamp=_now_iso(),
    )
