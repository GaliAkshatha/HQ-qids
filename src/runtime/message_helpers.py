"""
src/runtime/message_helpers.py

Tiny shared helper so every worker constructs PipelineMessage instances
the same way (event_id/timestamp generation), instead of duplicating
this in all 6 services. Also holds the one contract that needs custom
reconstruction from a plain dict: RoutingDecision nests a QuantumResult
dataclass, which dataclasses.asdict() flattens into a plain dict on
serialization but does not automatically restore on construction.
Every other contract used on the wire (DetectionResult, HybridDecision,
RiskAssessment, DefenseResult) is flat and reconstructs directly via
Contract(**data).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.contracts import PipelineMessage, QuantumResult, RoutingDecision


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_message(
    event_type: str,
    correlation_id: str,
    incident_id: str,
    payload: Dict[str, Any],
    causation_id: Optional[str] = None,
    retry_count: int = 0,
) -> PipelineMessage:
    return PipelineMessage(
        event_id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        causation_id=causation_id,
        incident_id=incident_id,
        event_type=event_type,
        timestamp=now_iso(),
        payload=payload,
        retry_count=retry_count,
    )


def routing_decision_from_dict(data: Dict[str, Any]) -> RoutingDecision:
    data = dict(data)
    if data.get("quantum_result") is not None:
        data["quantum_result"] = QuantumResult(**data["quantum_result"])
    return RoutingDecision(**data)
