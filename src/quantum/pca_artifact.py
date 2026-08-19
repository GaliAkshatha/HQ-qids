"""
src/quantum/pca_artifact.py

The quantum branch's dimensionality reduction step. Takes Phase 1's
scaled 41-dimensional output (Data/processed/X_train.npy) and reduces it
to 4 dimensions, matching the qubit count QSVM/VQC use.

This is intentionally shared and idempotent (get_or_fit_quantum_pca):
whichever of train_qsvm.py / train_vqc.py runs first fits and persists
artifacts/preprocessing/quantum_pca.joblib; the other reuses it. Since
sklearn's default PCA solver is deterministic for this data shape, fitting
it twice independently would produce numerically identical components
anyway -- sharing it just guarantees exactly one artifact file and avoids
doing the fit twice.

Separate from src/preprocessing/pca_reduction.py (the original, legacy,
non-persisting script) and separate from src/preprocessing/
classical_pipeline.py (Phase 1, no PCA at all, by design).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

QUANTUM_PCA_FILENAME = "quantum_pca.joblib"


def fit_quantum_pca(X_train_scaled: np.ndarray, n_components: int = 4, random_state: int = 42) -> PCA:
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train_scaled)
    return pca


def save_quantum_pca(pca: PCA, preprocessing_dir: str | Path) -> Path:
    preprocessing_dir = Path(preprocessing_dir)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    path = preprocessing_dir / QUANTUM_PCA_FILENAME
    joblib.dump(pca, path)
    return path


def load_quantum_pca(preprocessing_dir: str | Path) -> PCA:
    path = Path(preprocessing_dir) / QUANTUM_PCA_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted quantum PCA at {path}. Run train_qsvm.py or "
            "train_vqc.py first -- either one fits and persists it."
        )
    return joblib.load(path)


def get_or_fit_quantum_pca(
    X_train_scaled: np.ndarray,
    preprocessing_dir: str | Path,
    n_components: int = 4,
    random_state: int = 42,
) -> PCA:
    """Idempotent: reuse the persisted PCA if present, else fit and persist it."""
    preprocessing_dir = Path(preprocessing_dir)
    path = preprocessing_dir / QUANTUM_PCA_FILENAME
    if path.exists():
        return joblib.load(path)
    pca = fit_quantum_pca(X_train_scaled, n_components=n_components, random_state=random_state)
    save_quantum_pca(pca, preprocessing_dir)
    return pca


def transform_to_quantum_features(X_scaled: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Reduce Phase 1 scaled feature vector(s) to the quantum branch's
    4-dimensional representation. Accepts a single sample (1D, shape
    (41,)) or a batch (2D, shape (n, 41)) and always returns 2D.
    """
    X_scaled = np.asarray(X_scaled, dtype=np.float64)
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)
    return pca.transform(X_scaled)
