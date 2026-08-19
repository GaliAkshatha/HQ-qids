"""
src/detection/ensemble_detector.py

Concrete classical detector: Random Forest + XGBoost (soft-voted supervised
prediction) + Isolation Forest (unsupervised anomaly score), producing the
existing DetectionResult contract unchanged.

Combination rule (approved):
    P_attack = (P_RF_attack + P_XGB_attack) / 2
    P_normal = (P_RF_normal + P_XGB_normal) / 2
    classical_prediction  = argmax(averaged probabilities)
    classical_confidence  = probability of the predicted class
    model_disagreement    = abs(P_RF_attack - P_XGB_attack)

Individual RF/XGBoost attack-probabilities and the raw Isolation Forest
score are stored in DetectionResult.metadata for later explainability
(SHAP, dashboard) without changing the contract schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

from src.contracts import DetectionResult
from src.detection.base import ClassicalDetector
from src.detection.if_normalization import IsolationForestNormalization
from src.preprocessing.classical_pipeline import (
    PreprocessingArtifacts,
    load_preprocessing_artifacts,
    transform_batch,
    transform_sample,
)

CLASS_LABELS = ("normal", "attack")  # index 0, index 1 -- matches y encoding


class EnsembleClassicalDetector(ClassicalDetector):
    def __init__(
        self,
        rf_model: RandomForestClassifier,
        xgb_model: XGBClassifier,
        if_model: IsolationForest,
        if_normalization: IsolationForestNormalization,
        preprocessing: PreprocessingArtifacts,
    ) -> None:
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.if_model = if_model
        self.if_normalization = if_normalization
        self.preprocessing = preprocessing

        # sklearn / xgboost both sort classes ascending -> [0, 1] here,
        # which matches CLASS_LABELS order. Fail loudly if that ever
        # stops being true instead of silently mislabeling predictions.
        for name, model in (("random_forest", rf_model), ("xgboost", xgb_model)):
            classes = list(getattr(model, "classes_", [0, 1]))
            if classes != [0, 1]:
                raise ValueError(
                    f"{name} has unexpected classes_ {classes}; "
                    "expected [0, 1] (normal, attack)."
                )

    @classmethod
    def load(
        cls,
        models_dir: str | Path,
        preprocessing_dir: str | Path,
    ) -> "EnsembleClassicalDetector":
        """
        Build a detector purely from persisted artifacts on disk -- the
        entry point every other service (quantum router, dashboard, RAG)
        should use, so nothing depends on an in-memory training session.
        """
        models_dir = Path(models_dir)
        preprocessing_dir = Path(preprocessing_dir)

        rf_model = joblib.load(models_dir / "random_forest.joblib")
        xgb_model = joblib.load(models_dir / "xgboost.joblib")
        if_model = joblib.load(models_dir / "isolation_forest.joblib")
        if_normalization = IsolationForestNormalization.load(
            preprocessing_dir / "isolation_forest_normalization.json"
        )
        preprocessing = load_preprocessing_artifacts(preprocessing_dir)

        return cls(
            rf_model=rf_model,
            xgb_model=xgb_model,
            if_model=if_model,
            if_normalization=if_normalization,
            preprocessing=preprocessing,
        )

    # -- single sample -----------------------------------------------------

    def detect(self, sample: Mapping[str, object], sample_id: Optional[str] = None) -> DetectionResult:
        X = transform_sample(sample, self.preprocessing)
        return self._detect_from_matrix(X, sample_ids=[sample_id or "sample_0"])[0]

    # -- vectorized batch ----------------------------------------------------

    def detect_batch(
        self,
        samples: pd.DataFrame,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[DetectionResult]:
        X = transform_batch(samples, self.preprocessing)
        ids = list(sample_ids) if sample_ids is not None else [f"sample_{i}" for i in range(len(samples))]
        return self._detect_from_matrix(X, sample_ids=ids)

    def detect_matrix(
        self,
        X: np.ndarray,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[DetectionResult]:
        """Detect directly from an already-scaled feature matrix (used by training/eval)."""
        ids = list(sample_ids) if sample_ids is not None else [f"sample_{i}" for i in range(X.shape[0])]
        return self._detect_from_matrix(X, sample_ids=ids)

    # -- shared core ---------------------------------------------------------

    def _detect_from_matrix(self, X: np.ndarray, sample_ids: Sequence[str]) -> List[DetectionResult]:
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        avg_proba = (rf_proba + xgb_proba) / 2.0

        predicted_idx = np.argmax(avg_proba, axis=1)
        confidence = avg_proba[np.arange(len(avg_proba)), predicted_idx]
        disagreement = np.abs(rf_proba[:, 1] - xgb_proba[:, 1])

        raw_if_scores = self.if_model.score_samples(X)
        anomaly_scores = self.if_normalization.normalize(raw_if_scores)

        results: List[DetectionResult] = []
        for i, sample_id in enumerate(sample_ids):
            results.append(
                DetectionResult(
                    sample_id=str(sample_id),
                    classical_prediction=CLASS_LABELS[predicted_idx[i]],
                    classical_confidence=float(confidence[i]),
                    class_probabilities={
                        "normal": float(avg_proba[i, 0]),
                        "attack": float(avg_proba[i, 1]),
                    },
                    anomaly_score=float(anomaly_scores[i]),
                    model_disagreement=float(disagreement[i]),
                    metadata={
                        "rf_probability_attack": float(rf_proba[i, 1]),
                        "xgb_probability_attack": float(xgb_proba[i, 1]),
                        "if_raw_score": float(raw_if_scores[i]),
                    },
                )
            )
        return results
