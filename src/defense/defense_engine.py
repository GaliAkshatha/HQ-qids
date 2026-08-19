"""
src/defense/defense_engine.py

Orchestrates policy -> safety validation -> execution -> independent
verification -> recovery (only if verification failed) -> DefenseResult +
audit record + metrics. Contains no action-selection, validation,
execution, verification, or recovery logic of its own -- each of those
lives exactly once, in its own module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.contracts import DefenseResult, DetectionResult, HybridDecision, RiskAssessment
from src.defense import action_catalog as ac
from src.defense import verification as ver
from src.defense.audit import DefenseAuditRecord, log_audit_record
from src.defense.defense_policy import DefensePolicyConfig
from src.defense.executor import DefenseExecutor
from src.defense.metrics import DefenseMetrics
from src.defense.recovery import attempt_recovery
from src.defense.safety_validator import DefenseSafetyValidator
from src.defense.simulated_executor import SimulatedDefenseExecutor
from src.defense.simulated_state import SimulatedNetworkState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DefenseEngine:
    def __init__(
        self,
        policy: Optional[DefensePolicyConfig] = None,
        state: Optional[SimulatedNetworkState] = None,
        executor: Optional[DefenseExecutor] = None,
        validator: Optional[DefenseSafetyValidator] = None,
        metrics: Optional[DefenseMetrics] = None,
    ) -> None:
        self.policy = policy or DefensePolicyConfig.load()
        self.state = state or SimulatedNetworkState()
        self.executor = executor or SimulatedDefenseExecutor(self.state)
        self.validator = validator or DefenseSafetyValidator(self.policy, self.state)
        self.metrics = metrics or DefenseMetrics()

    def process(
        self,
        detection_result: DetectionResult,
        hybrid_decision: HybridDecision,
        risk_assessment: RiskAssessment,
    ) -> DefenseResult:
        target = risk_assessment.sample_id
        risk_level = risk_assessment.risk_level
        decision_status = hybrid_decision.decision_status

        selected_action = self.policy.select_action(risk_level, decision_status)
        policy_reason = self.policy.policy_reason(risk_level, decision_status, selected_action)

        validation = self.validator.validate(
            selected_action, target, risk_level, decision_status,
            detection_result, hybrid_decision, risk_assessment,
        )

        if not validation.allowed:
            return self._finalize_rejected(detection_result, risk_assessment, selected_action, target, policy_reason, validation)

        exec_result = self.executor.execute(selected_action, target)
        verification = ver.verify(self.state, selected_action, target)

        recovery_result = None
        if not verification.verified:
            spec = ac.get_action_spec(selected_action)
            recovery_result = attempt_recovery(
                executor=self.executor, state=self.state, action_type=selected_action, target=target,
                pre_action_snapshot=exec_result.before_state,
                max_retries=self.policy.max_recovery_retries,
                rollback_on_failure=self.policy.rollback_on_failure,
                action_reversible=spec.reversible,
            )

        return self._finalize_executed(
            detection_result, risk_assessment, selected_action, target, policy_reason,
            validation, exec_result, verification, recovery_result,
        )

    # ---- finalization paths -------------------------------------------------

    def _finalize_rejected(self, detection_result, risk_assessment, selected_action, target, policy_reason, validation) -> DefenseResult:
        already_active = validation.checks.get("not_already_active") is False
        health_status = "HEALTHY" if already_active else "UNHEALTHY"

        defense_result = DefenseResult(
            sample_id=detection_result.sample_id,
            severity=risk_assessment.risk_level,
            risk_score=risk_assessment.risk_score,
            action=selected_action,
            action_status="REJECTED",
            recovery_status="NOT_ATTEMPTED",
            health_status=health_status,
            rollback_available=False,
            timestamp=_now_iso(),
        )

        self.metrics.record(
            selected_action=selected_action, action_status="REJECTED", health_status=health_status,
            initial_verification_failed=False, recovery_attempted=False,
            recovery_status="NOT_ATTEMPTED", rollback_attempted=False,
        )

        record = DefenseAuditRecord(
            sample_id=detection_result.sample_id, risk_level=risk_assessment.risk_level, risk_score=risk_assessment.risk_score,
            selected_action=selected_action, target=target, policy_reason=policy_reason,
            validation_result={"allowed": validation.allowed, "reason": validation.reason, "checks": validation.checks},
            execution_result={}, verification_result={}, recovery_attempted=False, recovery_result=None,
            timestamp=defense_result.timestamp,
        )
        log_audit_record(record)
        return defense_result

    def _finalize_executed(
        self, detection_result, risk_assessment, selected_action, target, policy_reason,
        validation, exec_result, verification, recovery_result,
    ) -> DefenseResult:
        spec = ac.get_action_spec(selected_action)

        if verification.verified:
            action_status, health_status, recovery_status = "EXECUTED", "HEALTHY", "NOT_ATTEMPTED"
        elif recovery_result is not None and recovery_result.self_healed:
            action_status, health_status, recovery_status = "EXECUTED", "HEALTHY", "SUCCESS"
        else:
            action_status, health_status, recovery_status = "FAILED", "UNHEALTHY", "FAILED"

        defense_result = DefenseResult(
            sample_id=detection_result.sample_id,
            severity=risk_assessment.risk_level,
            risk_score=risk_assessment.risk_score,
            action=selected_action,
            action_status=action_status,
            recovery_status=recovery_status,
            health_status=health_status,
            rollback_available=spec.reversible,
            timestamp=_now_iso(),
        )

        self.metrics.record(
            selected_action=selected_action, action_status=action_status, health_status=health_status,
            initial_verification_failed=not verification.verified,
            recovery_attempted=recovery_result is not None,
            recovery_status=recovery_result.recovery_status if recovery_result else "NOT_ATTEMPTED",
            rollback_attempted=bool(recovery_result and recovery_result.rollback_attempted),
        )

        record = DefenseAuditRecord(
            sample_id=detection_result.sample_id, risk_level=risk_assessment.risk_level, risk_score=risk_assessment.risk_score,
            selected_action=selected_action, target=target, policy_reason=policy_reason,
            validation_result={"allowed": validation.allowed, "reason": validation.reason, "checks": validation.checks},
            execution_result={
                "succeeded": exec_result.succeeded, "error": exec_result.error,
                "before_state": vars(exec_result.before_state), "after_state": vars(exec_result.after_state),
            },
            verification_result={"verified": verification.verified, "checked_fields": verification.checked_fields, "reason": verification.reason},
            recovery_attempted=recovery_result is not None,
            recovery_result=(
                {
                    "retry_count": recovery_result.retry_count, "retry_succeeded": recovery_result.retry_succeeded,
                    "rollback_attempted": recovery_result.rollback_attempted, "rollback_succeeded": recovery_result.rollback_succeeded,
                    "self_healed": recovery_result.self_healed, "recovery_status": recovery_result.recovery_status,
                    "log": recovery_result.log,
                }
                if recovery_result else None
            ),
            timestamp=defense_result.timestamp,
        )
        log_audit_record(record)
        return defense_result
