"""
src/quantum/vqc_verifier.py

Unlike QSVM, VQC is a self-contained trained model -- no reference
training set needs to be persisted or re-evaluated at inference time.
A single forward pass through the trained circuit is enough, which is why
VQC inference is fast relative to QSVM (see Phase 2 plan timing notes).

Persisted via qiskit-machine-learning's to_dill()/from_dill(), the
current non-deprecated serialization API (legacy save_vqc_model.py uses
raw pickle; this module deliberately does not).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from qiskit_machine_learning.algorithms import VQC

from src.contracts import QuantumResult
from src.quantum.base import QuantumVerifier
from src.quantum.pca_artifact import load_quantum_pca, transform_to_quantum_features

CLASS_LABELS = ("normal", "attack")


class VQCVerifier(QuantumVerifier):
    model_name = "VQC"

    def __init__(self, vqc_model: VQC, circuit_metadata: dict, quantum_pca) -> None:
        self.vqc_model = vqc_model
        self.circuit_metadata = circuit_metadata
        self.quantum_pca = quantum_pca

    @classmethod
    def load(cls, models_dir: str | Path, preprocessing_dir: str | Path) -> "VQCVerifier":
        models_dir = Path(models_dir)
        vqc_model = VQC.from_dill(str(models_dir / "vqc_model.dill"))

        with open(models_dir / "feature_map_config.json") as f:
            feature_map_config = json.load(f)
        with open(models_dir / "ansatz_config.json") as f:
            ansatz_config = json.load(f)
        with open(models_dir / "optimizer_config.json") as f:
            optimizer_config = json.load(f)

        circuit_metadata = {
            "feature_map": feature_map_config,
            "ansatz": ansatz_config,
            "optimizer": optimizer_config,
        }
        quantum_pca = load_quantum_pca(preprocessing_dir)
        return cls(vqc_model=vqc_model, circuit_metadata=circuit_metadata, quantum_pca=quantum_pca)

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
            proba = self.vqc_model.predict_proba(X_pca)
            proba = np.asarray(proba)
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
                        circuit_metadata=self.circuit_metadata,
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
                    circuit_metadata=self.circuit_metadata,
                    inference_time_ms=per_sample_ms,
                )
                for sample_id in ids
            ]
