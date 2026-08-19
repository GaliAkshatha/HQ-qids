"""
src/quantum/subsampling.py

Matches the exact subsampling convention already used by
src/models/qsvm_model.py / vqc_model.py / save_qsvm_model.py /
save_vqc_model.py: a stratified train_test_split, keeping only the
`train_size` portion, discarding the rest. Kept as one shared helper so
train_qsvm.py and train_vqc.py don't each reimplement it slightly
differently.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=random_state
    )
    return X_sub, y_sub
