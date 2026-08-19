from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


VALID_QUANTUM_MODELS = {"QSVM", "VQC"}
VALID_STATUSES = {"success", "failed"}


@dataclass
class QuantumResult:
    """
    Output contract for the Quantum Verification Engine (Phase 2).

    Produced by QSVMVerifier / VQCVerifier. Consumed by the future Quantum
    Router (Phase 3) and folded into HybridDecision's existing quantum_*
    fields (Phase 4) -- this contract does not change HybridDecision,
    DetectionResult, or DefenseResult.

    status/error exist specifically so a failed quantum call is a valid,
    inspectable object rather than an uncaught exception -- this is what
    lets Phase 3's circuit breaker key off a clean signal instead of
    needing to wrap every call site in its own try/except.
    """

    sample_id: str
    quantum_model: str  # "QSVM" | "VQC"
    status: str  # "success" | "failed"

    quantum_prediction: Optional[str] = None
    quantum_confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None

    circuit_metadata: Dict[str, Any] = field(default_factory=dict)
    inference_time_ms: Optional[float] = None

    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = "1.0"

    def __post_init__(self):
        if self.quantum_model not in VALID_QUANTUM_MODELS:
            raise ValueError(
                f"quantum_model must be one of {VALID_QUANTUM_MODELS}. "
                f"Received: {self.quantum_model}"
            )

        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"status must be one of {VALID_STATUSES}. Received: {self.status}"
            )

        if self.status == "success":
            if self.quantum_prediction is None:
                raise ValueError("quantum_prediction is required when status='success'")
            if self.quantum_confidence is None:
                raise ValueError("quantum_confidence is required when status='success'")
            self._validate_probability(self.quantum_confidence, "quantum_confidence")
            if self.class_probabilities:
                for label, probability in self.class_probabilities.items():
                    self._validate_probability(probability, f"class_probabilities[{label}]")

        if self.status == "failed" and self.error is None:
            raise ValueError("error should be populated when status='failed'")

    @staticmethod
    def _validate_probability(value: float, name: str):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. Received: {value}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
