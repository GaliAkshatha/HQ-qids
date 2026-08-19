"""
src/quantum/qsvm_verifier.py

QSVM is a kernel method: scoring a new sample requires evaluating the
quantum kernel between it and the persisted reference training vectors,
then handing that row to the trained (classical) SVC. There is no
self-contained "model" the way there is for VQC -- the reference vectors
are part of the artifact, not just training-time scaffolding.

Measured cost (see Phase 2 plan): ~6.3ms per kernel pair on this
environment's simulator. At a 300-point reference set, a single verify()
call costs ~1.9s. verify_batch() evaluates the whole batch's kernel row in
one call instead of looping verify() N times -- mathematically identical
result, but avoids redundant per-sample overhead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import joblib
import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

from src.contracts import QuantumResult
from src.quantum.base import QuantumVerifier
from src.quantum.feature_maps import build_feature_map
from src.quantum.pca_artifact import load_quantum_pca, transform_to_quantum_features

CLASS_LABELS = ("normal", "attack")


class QSVMVerifier(QuantumVerifier):
    model_name = "QSVM"

    def __init__(
        self,
        svc_model: SVC,
        reference_vectors: np.ndarray,
        feature_map_config: dict,
        quantum_pca,
    ) -> None:
        self.svc_model = svc_model
        self.reference_vectors = reference_vectors
        self.feature_map_config = feature_map_config
        self.feature_map = build_feature_map(feature_map_config)
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        self.quantum_pca = quantum_pca

        classes = list(getattr(svc_model, "classes_", [0, 1]))
        if classes != [0, 1]:
            raise ValueError(f"QSVM SVC has unexpected classes_ {classes}; expected [0, 1].")

    @classmethod
    def load(cls, models_dir: str | Path, preprocessing_dir: str | Path) -> "QSVMVerifier":
        models_dir = Path(models_dir)
        svc_model = joblib.load(models_dir / "svc_model.joblib")
        reference_vectors = np.load(models_dir / "reference_vectors.npy")
        with open(models_dir / "feature_map_config.json") as f:
            feature_map_config = json.load(f)
        quantum_pca = load_quantum_pca(preprocessing_dir)
        return cls(
            svc_model=svc_model,
            reference_vectors=reference_vectors,
            feature_map_config=feature_map_config,
            quantum_pca=quantum_pca,
        )

    def verify(self, scaled_features: np.ndarray, sample_id: str) -> QuantumResult:
        results = self.verify_batch(np.asarray(scaled_features).reshape(1, -1), [sample_id])
        return results[0]

    def verify_batch(
        self,
        scaled_features_matrix: np.ndarray,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> List[QuantumResult]:
        ids = list(sample_ids) if sample_ids is not None else [f"sample_{i}" for i in range(len(scaled_features_matrix))]
        t0 = time.perf_counter()
        try:
            X_pca = transform_to_quantum_features(scaled_features_matrix, self.quantum_pca)
            kernel_rows = self.kernel.evaluate(x_vec=X_pca, y_vec=self.reference_vectors)
            proba = self.svc_model.predict_proba(kernel_rows)
            total_ms = (time.perf_counter() - t0) * 1000.0
            per_sample_ms = total_ms / len(ids)

            results = []
            for i, sample_id in enumerate(ids):
                pred_idx = int(np.argmax(proba[i]))
                results.append(
                    QuantumResult(
                        sample_id=str(sample_id),
                        quantum_model=self.model_name,
                        status="success",
                        quantum_prediction=CLASS_LABELS[pred_idx],
                        quantum_confidence=float(proba[i, pred_idx]),
                        class_probabilities={
                            "normal": float(proba[i, 0]),
                            "attack": float(proba[i, 1]),
                        },
                        circuit_metadata=self.feature_map_config,
                        inference_time_ms=per_sample_ms,
                    )
                )
            return results
        except Exception as e:  # noqa: BLE001 -- deliberate: never let quantum failure raise
            total_ms = (time.perf_counter() - t0) * 1000.0
            per_sample_ms = total_ms / max(len(ids), 1)
            return [
                QuantumResult(
                    sample_id=str(sample_id),
                    quantum_model=self.model_name,
                    status="failed",
                    error=str(e),
                    circuit_metadata=self.feature_map_config,
                    inference_time_ms=per_sample_ms,
                )
                for sample_id in ids
            ]
