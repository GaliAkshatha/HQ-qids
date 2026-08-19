"""
src/quantum/base.py

Common interface for QSVM and VQC verifiers. Mirrors the shape of
src/detection/base.py's ClassicalDetector so a future router can treat
classical and quantum detectors symmetrically -- but this module builds
only the verifiers themselves. No router, no auto-invocation of both
models, no circuit breaker: those are later phases.

verify() takes Phase 1's scaled 41-dimensional feature vector directly
(NOT a DetectionResult -- that contract is not extended with a raw
feature vector). Whatever calls this in Phase 3 is responsible for
threading the feature vector through alongside the DetectionResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from src.contracts import QuantumResult


class QuantumVerifier(ABC):
    model_name: str

    @abstractmethod
    def verify(self, scaled_features: np.ndarray, sample_id: str) -> QuantumResult:
        """
        Run quantum verification on a single Phase-1-scaled (41-dim)
        feature vector. Must never raise -- failures are represented as
        QuantumResult(status="failed", error=...).
        """
        raise NotImplementedError

    def verify_batch(
        self,
        scaled_features_matrix: np.ndarray,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[QuantumResult]:
        """
        Default (non-vectorized) batch implementation: loops over verify().
        QSVMVerifier overrides this with a single batched kernel
        evaluation, which is substantially cheaper than N independent
        calls -- see its docstring for why that matters here.
        """
        ids = list(sample_ids) if sample_ids is not None else [f"sample_{i}" for i in range(len(scaled_features_matrix))]
        return [self.verify(scaled_features_matrix[i], ids[i]) for i in range(len(scaled_features_matrix))]
