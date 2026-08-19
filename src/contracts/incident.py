"""
src/contracts/incident.py

IncidentEvent and IncidentSnapshot -- the Phase 6 contracts. Same
validation/to_dict/schema_version approach as every other contract in
this project. Additive only: DetectionResult, QuantumResult,
RoutingDecision, HybridDecision, RiskAssessment, DefenseResult are all
unmodified.

State and event-type vocabularies are centralized here as the single
source of truth (imported by src/incident/*, never re-typed as string
literals elsewhere) -- this is what "keep event types centralized as
constants" means concretely.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Incident lifecycle states
# ---------------------------------------------------------------------------

DETECTED = "DETECTED"
ASSESSING = "ASSESSING"
VERIFYING = "VERIFYING"
MITIGATING = "MITIGATING"
RECOVERY = "RECOVERY"
RESOLVED = "RESOLVED"
ESCALATED = "ESCALATED"

INCIDENT_STATES = {DETECTED, ASSESSING, VERIFYING, MITIGATING, RECOVERY, RESOLVED, ESCALATED}
TERMINAL_STATES = {RESOLVED, ESCALATED}

# Approved transition table (Decision 6). Invalid edges must fail
# explicitly -- see src/incident/incident_state.py's StateMachine.
VALID_TRANSITIONS: Dict[str, set] = {
    DETECTED: {ASSESSING},
    ASSESSING: {VERIFYING, MITIGATING},
    VERIFYING: {MITIGATING},
    MITIGATING: {RECOVERY, RESOLVED, ESCALATED},
    RECOVERY: {RESOLVED, ESCALATED},
    RESOLVED: set(),
    ESCALATED: set(),
}

# ---------------------------------------------------------------------------
# Event types -- 15 lifecycle events + IDEMPOTENT_SKIP = 16 today. This set
# is expected to grow (Phase 7+); nothing in this project should assume
# "exactly 15" or "exactly 16" -- membership in EVENT_TYPES is the only
# thing validated, not a fixed count.
# ---------------------------------------------------------------------------

DETECTION_CREATED = "DETECTION_CREATED"
QUANTUM_ROUTING_REQUESTED = "QUANTUM_ROUTING_REQUESTED"
QUANTUM_VERIFICATION_COMPLETED = "QUANTUM_VERIFICATION_COMPLETED"
QUANTUM_VERIFICATION_FAILED = "QUANTUM_VERIFICATION_FAILED"
HYBRID_DECISION_CREATED = "HYBRID_DECISION_CREATED"
RISK_ASSESSED = "RISK_ASSESSED"
DEFENSE_ACTION_SELECTED = "DEFENSE_ACTION_SELECTED"
DEFENSE_ACTION_EXECUTED = "DEFENSE_ACTION_EXECUTED"
DEFENSE_VERIFICATION_FAILED = "DEFENSE_VERIFICATION_FAILED"
RECOVERY_STARTED = "RECOVERY_STARTED"
RECOVERY_SUCCEEDED = "RECOVERY_SUCCEEDED"
RECOVERY_FAILED = "RECOVERY_FAILED"
ROLLBACK_EXECUTED = "ROLLBACK_EXECUTED"
INCIDENT_RESOLVED = "INCIDENT_RESOLVED"
INCIDENT_ESCALATED = "INCIDENT_ESCALATED"
IDEMPOTENT_SKIP = "IDEMPOTENT_SKIP"

EVENT_TYPES = {
    DETECTION_CREATED, QUANTUM_ROUTING_REQUESTED, QUANTUM_VERIFICATION_COMPLETED,
    QUANTUM_VERIFICATION_FAILED, HYBRID_DECISION_CREATED, RISK_ASSESSED,
    DEFENSE_ACTION_SELECTED, DEFENSE_ACTION_EXECUTED, DEFENSE_VERIFICATION_FAILED,
    RECOVERY_STARTED, RECOVERY_SUCCEEDED, RECOVERY_FAILED, ROLLBACK_EXECUTED,
    INCIDENT_RESOLVED, INCIDENT_ESCALATED, IDEMPOTENT_SKIP,
}


@dataclass
class IncidentEvent:
    """
    One immutable, append-only fact in an incident's history. Never
    mutated after creation. `payload` carries the relevant upstream
    contract's to_dict() (e.g. DetectionResult, RiskAssessment) so the
    event history is independently auditable without needing to re-fetch
    anything from elsewhere.
    """

    event_id: str
    correlation_id: str
    incident_id: str
    event_type: str
    previous_state: Optional[str]
    new_state: Optional[str]
    timestamp: str
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event_type: '{self.event_type}'. Known types: {sorted(EVENT_TYPES)}")
        if self.previous_state is not None and self.previous_state not in INCIDENT_STATES:
            raise ValueError(f"Unknown previous_state: '{self.previous_state}'")
        if self.new_state is not None and self.new_state not in INCIDENT_STATES:
            raise ValueError(f"Unknown new_state: '{self.new_state}'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentEvent":
        return cls(**data)


@dataclass
class IncidentSnapshot:
    """
    Fast-lookup materialized view of an incident's current state --
    always reconstructable from IncidentEvent history alone (see
    src/incident/event_store.py's replay logic). Never the sole source of
    truth; the event log is.
    """

    incident_id: str
    correlation_id: str
    current_state: str
    created_at: str
    updated_at: str
    event_ids: List[str] = field(default_factory=list)
    escalated: bool = False
    escalation_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.current_state not in INCIDENT_STATES:
            raise ValueError(f"Unknown current_state: '{self.current_state}'. Known states: {sorted(INCIDENT_STATES)}")
        if self.escalated and not self.escalation_reasons:
            raise ValueError("escalation_reasons must be non-empty when escalated=True")

    @property
    def is_terminal(self) -> bool:
        return self.current_state in TERMINAL_STATES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentSnapshot":
        return cls(**data)
