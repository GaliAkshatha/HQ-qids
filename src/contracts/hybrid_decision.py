from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


VALID_DECISION_STATUSES = {
    "confirmed",
    "uncertain",
    "normal",
}


@dataclass
class HybridDecision:
    """
    Output contract for Part 2: Hybrid Quantum Intelligence.

    Consumed by Part 3 and Part 4.
    """

    sample_id: str

    final_prediction: str
    final_confidence: float

    quantum_used: bool

    quantum_model: Optional[str] = None
    quantum_prediction: Optional[str] = None
    quantum_confidence: Optional[float] = None

    verification_reason: List[str] = field(default_factory=list)

    decision_status: str = "uncertain"

    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = "1.0"

    def __post_init__(self):
        self._validate_probability(
            self.final_confidence,
            "final_confidence"
        )

        if self.quantum_confidence is not None:
            self._validate_probability(
                self.quantum_confidence,
                "quantum_confidence"
            )

        if self.decision_status not in VALID_DECISION_STATUSES:
            raise ValueError(
                f"decision_status must be one of "
                f"{VALID_DECISION_STATUSES}. "
                f"Received: {self.decision_status}"
            )

        if self.quantum_used and self.quantum_model is None:
            raise ValueError(
                "quantum_model must be provided when quantum_used=True"
            )

    @staticmethod
    def _validate_probability(value: float, name: str):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1. Received: {value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)