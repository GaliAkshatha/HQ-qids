from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .quantum_result import QuantumResult


VALID_DECISION_STATUSES = {"not_invoked", "pending", "success", "fallback"}
VALID_CIRCUIT_STATES = {"CLOSED", "OPEN", "HALF_OPEN"}


@dataclass
class RoutingDecision:
    """
    Output contract for the Quantum Router (Phase 3).

    A RoutingDecision has a lifecycle: route() returns one immediately in
    "pending" status (if a quantum job was submitted) without waiting for
    it to finish -- it must never claim a quantum_result exists before the
    job actually completes. Calling get_result()/route_and_wait() later
    produces an updated RoutingDecision (same sample_id) with
    decision_status advanced to "success" or "fallback".

    Consumed by the future Phase 4 HybridDecision assembly via
    reason_codes -> HybridDecision.verification_reason and
    quantum_result's fields -> HybridDecision.quantum_*. This contract
    does not change DetectionResult, QuantumResult, HybridDecision, or
    DefenseResult.
    """

    sample_id: str
    decision_status: str  # "not_invoked" | "pending" | "success" | "fallback"

    should_invoke_quantum: bool
    reason_codes: List[str] = field(default_factory=list)
    signal_values: Dict[str, float] = field(default_factory=dict)
    policy_thresholds: Dict[str, float] = field(default_factory=dict)

    quantum_backend: Optional[str] = None  # "QSVM" | "VQC" | None
    circuit_breaker_state: str = "CLOSED"
    quantum_available: bool = True

    quantum_attempted: bool = False
    quantum_result: Optional[QuantumResult] = None

    fallback_used: bool = False
    fallback_reason: Optional[str] = None  # "circuit_open" | "timeout" | "retries_exhausted"

    job_id: Optional[str] = None

    routing_latency_ms: Optional[float] = None
    queue_wait_time_ms: Optional[float] = None
    quantum_execution_time_ms: Optional[float] = None
    total_quantum_job_time_ms: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.decision_status not in VALID_DECISION_STATUSES:
            raise ValueError(
                f"decision_status must be one of {VALID_DECISION_STATUSES}. "
                f"Received: {self.decision_status}"
            )
        if self.circuit_breaker_state not in VALID_CIRCUIT_STATES:
            raise ValueError(
                f"circuit_breaker_state must be one of {VALID_CIRCUIT_STATES}. "
                f"Received: {self.circuit_breaker_state}"
            )
        if self.decision_status == "pending" and self.quantum_result is not None:
            raise ValueError(
                "quantum_result must be None while decision_status='pending' -- "
                "a pending decision must not pretend a result exists yet."
            )
        if self.decision_status == "success" and self.quantum_result is None:
            raise ValueError("quantum_result is required when decision_status='success'")
        if self.fallback_used and self.fallback_reason is None:
            raise ValueError("fallback_reason is required when fallback_used=True")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
