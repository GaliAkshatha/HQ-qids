from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class DetectionResult:
    """
    Output contract for Part 1: Classical AI Detection Engine.

    This object is consumed by Part 2: Hybrid Quantum Intelligence.
    """

    sample_id: str
    classical_prediction: str
    classical_confidence: float
    class_probabilities: Dict[str, float]
    anomaly_score: float
    model_disagreement: float

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        self._validate_probability(
            self.classical_confidence,
            "classical_confidence"
        )

        self._validate_probability(
            self.anomaly_score,
            "anomaly_score"
        )

        self._validate_probability(
            self.model_disagreement,
            "model_disagreement"
        )

        for label, probability in self.class_probabilities.items():
            self._validate_probability(
                probability,
                f"class_probabilities[{label}]"
            )

    @staticmethod
    def _validate_probability(value: float, name: str):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1. Received: {value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)