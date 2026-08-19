"""
src/detection/base.py

Abstract interface every classical detector implementation must satisfy.
Kept deliberately small so the ensemble implementation can later be
swapped or extended (e.g. a different model set) without touching callers
in Phase 2+ (quantum router, hybrid decision, dashboard, RAG).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from src.contracts import DetectionResult


class ClassicalDetector(ABC):
    """Reusable interface for the classical detection stage."""

    @abstractmethod
    def detect(self, sample: Mapping[str, object], sample_id: Optional[str] = None) -> DetectionResult:
        """Run detection on a single raw traffic sample."""
        raise NotImplementedError

    def detect_batch(
        self,
        samples: pd.DataFrame,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[DetectionResult]:
        """
        Default (non-vectorized) batch implementation: loops over detect().
        Implementations should override this with a vectorized version when
        the underlying models support it -- see EnsembleClassicalDetector.
        """
        results = []
        records: Iterable[dict] = samples.to_dict(orient="records")
        for i, record in enumerate(records):
            sid = sample_ids[i] if sample_ids is not None else None
            results.append(self.detect(record, sample_id=sid))
        return results
