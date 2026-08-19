from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}  # matches DefenseResult.VALID_SEVERITIES exactly


@dataclass
class RiskAssessment:
    """
    Output contract for the Risk Engine (Phase 4).

    risk_level uses the exact same vocabulary as the existing (unmodified)
    DefenseResult.VALID_SEVERITIES, and risk_score matches its
    [0,1]-bounded risk_score field -- this is deliberate: DefenseResult
    already had these fields defined since Phase 0, sized for exactly this
    hand-off. Phase 4 does not touch DefenseResult; Phase 5 will read
    RiskAssessment.risk_level / risk_score to populate it.

    contributing_factors keeps THREAT EVIDENCE and SYSTEM UNCERTAINTY
    separately labeled (not just one flat dict) so it's always inspectable
    which kind of signal drove a given risk level -- fallback/uncertainty
    should never look like it was threat evidence after the fact.
    """

    sample_id: str
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    risk_score: float  # [0, 1]

    threat_evidence_score: float
    system_uncertainty_score: float

    reason_codes: List[str] = field(default_factory=list)
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    policy_thresholds: Dict[str, float] = field(default_factory=dict)

    floor_applied: Optional[str] = None  # e.g. "confirmed_attack_min_level"

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"risk_level must be one of {VALID_RISK_LEVELS}. Received: {self.risk_level}")
        self._validate_unit_interval(self.risk_score, "risk_score")
        self._validate_unit_interval(self.threat_evidence_score, "threat_evidence_score")
        self._validate_unit_interval(self.system_uncertainty_score, "system_uncertainty_score")

    @staticmethod
    def _validate_unit_interval(value: float, name: str):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. Received: {value}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
