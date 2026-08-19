"""
src/defense/audit.py

DefenseAuditRecord carries the full rich detail your spec asks for --
policy_reason, validation_result, execution_result, verification_result,
recovery_result -- deliberately kept OUT of DefenseResult, whose schema
is intentionally unchanged. Logged through the existing observability
infrastructure (src/observability/logging_config.py, Phase 1), not a new
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.observability.logging_config import get_logger, log_event

logger = get_logger("defense_engine")


@dataclass
class DefenseAuditRecord:
    sample_id: str
    risk_level: str
    risk_score: float
    selected_action: str
    target: str
    policy_reason: str
    validation_result: Dict[str, Any]
    execution_result: Dict[str, Any]
    verification_result: Dict[str, Any]
    recovery_attempted: bool
    recovery_result: Optional[Dict[str, Any]]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def log_audit_record(record: DefenseAuditRecord) -> None:
    log_event(
        logger, 20, "Defense event processed",
        sample_id=record.sample_id,
        risk_level=record.risk_level,
        risk_score=record.risk_score,
        selected_action=record.selected_action,
        target=record.target,
        policy_reason=record.policy_reason,
        validation_result=record.validation_result,
        execution_result=record.execution_result,
        verification_result=record.verification_result,
        recovery_attempted=record.recovery_attempted,
        recovery_result=record.recovery_result,
        timestamp=record.timestamp,
    )
