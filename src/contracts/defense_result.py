from dataclasses import dataclass, asdict
from typing import Dict, Any


VALID_SEVERITIES = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}


@dataclass
class DefenseResult:
    """
    Output contract for Part 3:
    Autonomous Defense & Self-Healing.

    Consumed primarily by Part 4.
    """

    sample_id: str

    severity: str
    risk_score: float

    action: str
    action_status: str

    recovery_status: str
    health_status: str

    rollback_available: bool
    timestamp: str

    schema_version: str = "1.0"

    def __post_init__(self):
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of "
                f"{VALID_SEVERITIES}. "
                f"Received: {self.severity}"
            )

        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                "risk_score must be between 0 and 1. "
                f"Received: {self.risk_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)