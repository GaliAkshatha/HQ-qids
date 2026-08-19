"""
src/hybrid/hybrid_engine.py

Combines DetectionResult (Phase 1) + a RESOLVED RoutingDecision (Phase 3,
which already embeds QuantumResult) into the existing HybridDecision
contract -- unmodified, per approved Decision 1.

decision_status semantics (approved, documented here as the single source
of truth for this interpretation -- the contract itself doesn't spell
this out):

    "normal"    = final prediction is normal, no unresolved conflict
    "confirmed" = final prediction is attack, AND the evidence backing it
                  is actually sufficient (quantum agreement, a quantum
                  override that cleared its own confidence bar, or a
                  classical-only call above classical_high_confidence_threshold)
    "uncertain" = an unresolved classical/quantum conflict exists (quantum
                  disagreed but didn't clear the override bar), OR a
                  classical-only/fallback attack call whose confidence
                  didn't clear classical_high_confidence_threshold.

    "confirmed" is NEVER assigned merely because final_prediction=="attack".
    "uncertain" can apply even when the retained classical confidence was
    independently high -- the conflict itself is what's being flagged.

decision_changed_by_quantum is stored in HybridDecision.evidence (not a
new top-level field, per approved Decision 3), computed as exactly:

    quantum_used AND (final_prediction != classical_prediction)
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.contracts import DetectionResult, HybridDecision, RoutingDecision
from src.hybrid.decision_policy import DecisionPolicyConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_hybrid_decision(
    detection_result: DetectionResult,
    routing_decision: RoutingDecision,
    policy: DecisionPolicyConfig,
) -> HybridDecision:
    if routing_decision.decision_status == "pending":
        raise ValueError(
            "build_hybrid_decision() requires a RESOLVED RoutingDecision "
            "(decision_status != 'pending'). Call route_and_wait() or poll "
            "get_result() to completion before building a HybridDecision."
        )
    if detection_result.sample_id != routing_decision.sample_id:
        raise ValueError(
            f"sample_id mismatch: DetectionResult={detection_result.sample_id!r} "
            f"vs RoutingDecision={routing_decision.sample_id!r}"
        )

    classical_prediction = detection_result.classical_prediction
    classical_confidence = detection_result.classical_confidence

    base_evidence = {
        "classical_confidence": classical_confidence,
        "anomaly_score": detection_result.anomaly_score,
        "model_disagreement": detection_result.model_disagreement,
    }

    # ---- Case 1: quantum not invoked -----------------------------------
    if routing_decision.decision_status == "not_invoked":
        final_prediction = classical_prediction
        final_confidence = classical_confidence
        verification_reason = list(routing_decision.reason_codes) + ["CLASSICAL_ONLY_HIGH_CONFIDENCE"]
        decision_status = _classical_only_status(final_prediction, classical_confidence, policy)
        evidence = {
            **base_evidence,
            "agreement": None,
            "decision_changed_by_quantum": False,
            "fallback_used": False,
        }
        return HybridDecision(
            sample_id=detection_result.sample_id,
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            quantum_used=False,
            quantum_model=None,
            quantum_prediction=None,
            quantum_confidence=None,
            verification_reason=verification_reason,
            decision_status=decision_status,
            evidence=evidence,
        )

    # ---- Case: quantum fallback (circuit open / timeout / retries) ----
    if routing_decision.decision_status == "fallback":
        final_prediction = classical_prediction
        final_confidence = classical_confidence
        verification_reason = list(routing_decision.reason_codes) + ["QUANTUM_FAILED_CLASSICAL_FALLBACK"]
        decision_status = _classical_only_status(final_prediction, classical_confidence, policy)
        evidence = {
            **base_evidence,
            "agreement": None,
            "decision_changed_by_quantum": False,
            "fallback_used": True,
            "fallback_reason": routing_decision.fallback_reason,
        }
        return HybridDecision(
            sample_id=detection_result.sample_id,
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            quantum_used=False,
            quantum_model=routing_decision.quantum_backend,  # informational: which backend was attempted
            quantum_prediction=None,
            quantum_confidence=None,
            verification_reason=verification_reason,
            decision_status=decision_status,
            evidence=evidence,
        )

    # ---- Case: quantum resolved successfully ---------------------------
    assert routing_decision.decision_status == "success"
    qr = routing_decision.quantum_result
    assert qr is not None and qr.status == "success"  # guaranteed by RoutingDecision's own validation

    quantum_prediction = qr.quantum_prediction
    quantum_confidence = qr.quantum_confidence
    agree = quantum_prediction == classical_prediction

    if agree:
        final_prediction = classical_prediction
        final_confidence = max(classical_confidence, quantum_confidence)
        verification_reason = list(routing_decision.reason_codes) + ["QUANTUM_CONFIRMED_CLASSICAL"]
        decision_status = "normal" if final_prediction == "normal" else "confirmed"
        evidence = {
            **base_evidence,
            "quantum_confidence": quantum_confidence,
            "agreement": True,
            "decision_changed_by_quantum": False,  # quantum_used True but final == classical
        }
    else:
        overrides = quantum_confidence >= policy.quantum_override_confidence_threshold
        if overrides:
            final_prediction = quantum_prediction
            final_confidence = quantum_confidence
            verification_reason = list(routing_decision.reason_codes) + ["QUANTUM_OVERRULED_CLASSICAL"]
            decision_status = "normal" if final_prediction == "normal" else "confirmed"
        else:
            final_prediction = classical_prediction
            final_confidence = classical_confidence
            verification_reason = list(routing_decision.reason_codes) + ["QUANTUM_UNCERTAIN_CLASSICAL_RETAINED"]
            decision_status = "uncertain"  # unresolved conflict, regardless of predicted class
        evidence = {
            **base_evidence,
            "quantum_confidence": quantum_confidence,
            "agreement": False,
            # exactly the approved formula: quantum_used AND (final != classical).
            # quantum_used is True throughout this whole branch, so this
            # reduces to the prediction comparison -- spelled out fully
            # rather than assuming, since that's the literal contract.
            "decision_changed_by_quantum": True and (final_prediction != classical_prediction),
            "quantum_override_applied": overrides,
            "quantum_override_threshold_used": policy.quantum_override_confidence_threshold,
        }

    return HybridDecision(
        sample_id=detection_result.sample_id,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        quantum_used=True,
        quantum_model=qr.quantum_model,
        quantum_prediction=quantum_prediction,
        quantum_confidence=quantum_confidence,
        verification_reason=verification_reason,
        decision_status=decision_status,
        evidence=evidence,
    )


def _classical_only_status(final_prediction: str, classical_confidence: float, policy: DecisionPolicyConfig) -> str:
    """Shared decision_status rule for the not_invoked and fallback cases:
    both rely purely on the classical result, so the same threshold applies."""
    if final_prediction == "normal":
        return "normal"
    return "confirmed" if classical_confidence >= policy.classical_high_confidence_threshold else "uncertain"
