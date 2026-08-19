"""
src/detection/if_normalization.py

IsolationForest.score_samples() returns a raw "normality" score where
LOWER means MORE anomalous -- the opposite convention of the
DetectionResult.anomaly_score contract, which requires a value in [0, 1]
where HIGHER means MORE anomalous.

This module fits that mapping once, at training time, and persists it
(if_score_min / if_score_max) so inference-time normalization is identical
to what was used during evaluation -- this is the "persisted Isolation
Forest normalization" refinement.

The bounds are computed from the FULL training split's raw scores (both
classes), not just the normal-only rows the model itself is fit on --
this keeps the observed range wide enough that attack traffic doesn't
all collapse to anomaly_score == 1.0 with no separation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestNormalization:
    score_min: float
    score_max: float

    def normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        span = self.score_max - self.score_min
        if span <= 0:
            # Degenerate case (e.g. tiny fixture data with near-constant
            # scores) -- avoid a divide-by-zero, fall back to a neutral
            # mid-range anomaly score instead of raising.
            return np.full_like(raw_scores, 0.5, dtype=np.float64)
        normalized = (self.score_max - raw_scores) / span
        return np.clip(normalized, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"score_min": self.score_min, "score_max": self.score_max}

    @classmethod
    def from_dict(cls, d: dict) -> "IsolationForestNormalization":
        return cls(score_min=d["score_min"], score_max=d["score_max"])

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestNormalization":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit_isolation_forest_normal_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[IsolationForest, IsolationForestNormalization]:
    """
    Fit IsolationForest on NORMAL-labeled training rows only (y_train == 0).

    Rationale: IsolationForest's premise is learning what "normal" looks
    like from a mostly-clean reference set. NSL-KDD's training split is
    close to 47% attack traffic -- fitting on the full mixed split would
    badly miscalibrate IsolationForest's default contamination assumption.
    Fitting on normal-only rows is the standard practice for this model
    used as an anomaly detector.

    Normalization bounds are then computed from scoring the FULL training
    split (both classes) with the fitted model, so the persisted range
    reflects realistic scores for both normal and attack traffic.
    """
    normal_mask = y_train == 0
    if_model = IsolationForest(
        n_estimators=n_estimators,
        random_state=random_state,
        contamination="auto",
        n_jobs=-1,
    )
    if_model.fit(X_train[normal_mask])

    raw_scores_full = if_model.score_samples(X_train)
    normalization = IsolationForestNormalization(
        score_min=float(raw_scores_full.min()),
        score_max=float(raw_scores_full.max()),
    )
    return if_model, normalization
