"""
src/agents/application_detector.py

Classical application-security detector. Trains on
ApplicationFeatureVector-derived rows (11 features), session-split (no
leakage), evaluated on a held-out test split with metrics reported
honestly.

ApplicationSecurityDetector wraps the CHOSEN trained classical model for
inference and emits a real, unmodified DetectionResult -- reusing the
existing Phase 1 contract's generic fields, NOT the NSL-KDD-specific
EnsembleClassicalDetector or transform_sample(). Legitimate contract
reuse, not feature forcing -- see docs/APPLICATION_SECURITY_MODEL_BOUNDARY.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.agents.application_dataset import FEATURE_NAMES, LabeledSample, to_matrix
from src.contracts import DetectionResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "models" / "application" / "classical"


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
    }


def train_classical_baselines(
    train: List[LabeledSample], val: List[LabeledSample], test: List[LabeledSample],
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = to_matrix(train)
    X_val, y_val = to_matrix(val)
    X_test, y_test = to_matrix(test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val else np.empty((0, len(FEATURE_NAMES)))
    X_test_scaled = scaler.transform(X_test) if X_test else np.empty((0, len(FEATURE_NAMES)))

    results: Dict[str, Any] = {}

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_test_pred = lr.predict(X_test_scaled) if X_test else []
    results["logistic_regression"] = {
        "test": evaluate(y_test, lr_test_pred) if X_test else None,
        "val": evaluate(y_val, lr.predict(X_val_scaled)) if X_val else None,
    }

    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_test_pred = rf.predict(X_test_scaled) if X_test else []
    results["random_forest"] = {
        "test": evaluate(y_test, rf_test_pred) if X_test else None,
        "val": evaluate(y_val, rf.predict(X_val_scaled)) if X_val else None,
    }

    joblib.dump(lr, model_dir / "logistic_regression.joblib")
    joblib.dump(rf, model_dir / "random_forest.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f)

    results["train_size"] = len(train)
    results["val_size"] = len(val)
    results["test_size"] = len(test)
    results["dataset_label"] = "AGENT_GENERATED_LABELED_DATA"

    return results


class ApplicationSecurityDetector:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, model_name: str = "random_forest") -> None:
        self.model = joblib.load(model_dir / f"{model_name}.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        with open(model_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)

    def detect(self, feature_vector, sample_id: str) -> DetectionResult:
        x = [[getattr(feature_vector, name) for name in self.feature_names]]
        x_scaled = self.scaler.transform(x)
        proba = self.model.predict_proba(x_scaled)[0]
        classes = list(self.model.classes_)
        prob_normal = float(proba[classes.index(0)]) if 0 in classes else 0.0
        prob_anomalous = float(proba[classes.index(1)]) if 1 in classes else 0.0

        prediction = "attack" if prob_anomalous >= prob_normal else "normal"
        confidence = max(prob_normal, prob_anomalous)

        return DetectionResult(
            sample_id=sample_id, classical_prediction=prediction, classical_confidence=confidence,
            class_probabilities={"normal": prob_normal, "attack": prob_anomalous},
            anomaly_score=prob_anomalous,
            model_disagreement=0.0,
            metadata={"source": "APPLICATION_SECURITY", "model_name": self.model.__class__.__name__},
        )
