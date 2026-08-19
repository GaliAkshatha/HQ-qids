"""
src/preprocessing/classical_pipeline.py

Reusable preprocessing for the CLASSICAL detection path (Random Forest +
XGBoost + Isolation Forest).

This is intentionally separate from src/training/train.py and
src/preprocessing/pca_reduction.py:

- src/training/train.py is left untouched. It is a top-level script that
  fits encoders/scaler in-memory but never persists them, so it cannot
  transform a new incoming sample at inference time. It remains as-is for
  the legacy SVM baseline (src/models/classical_model.py), which continues
  to operate on PCA'd features.

- This module persists every fitted transformer (label encoders + scaler)
  to artifacts/preprocessing/, and exposes transform_sample() so a single
  new traffic row can be turned into a model-ready feature vector the same
  way every time -- which is what a "reusable detector interface" needs.

- No PCA happens here. PCA remains scoped to the quantum branch, per the
  approved architecture.

Design note vs. train.py: this module fits the label encoders and scaler on
the TRAIN split only, after splitting -- train.py fits LabelEncoder on the
full dataframe before splitting. This avoids a (minor, but real) leakage
of test-set category information into the encoders, at the cost of not
being bit-for-bit identical to train.py's historical behaviour. Unseen
categories at inference/test time are mapped to an explicit UNKNOWN bucket
rather than raising, since NSL-KDD's test distribution is known to contain
service values not present in the training distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Schema -- must match src/training/train.py's `columns` list exactly, since
# both read the same raw NSL-KDD file.
# ---------------------------------------------------------------------------

RAW_COLUMNS: List[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL_COLUMNS: List[str] = ["protocol_type", "service", "flag"]

LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"

UNKNOWN_TOKEN = "__UNKNOWN__"


class SafeLabelEncoder:
    """
    A LabelEncoder-alike that does not raise on unseen categories at
    transform time. Unseen values are mapped to a dedicated UNKNOWN index
    appended after the fitted classes. Plain-attribute object so it is
    joblib-picklable without custom reducers.
    """

    def __init__(self) -> None:
        self.classes_: List[str] = []
        self._index: Dict[str, int] = {}

    def fit(self, values: Sequence[str]) -> "SafeLabelEncoder":
        self.classes_ = sorted(set(values))
        self._index = {cls: i for i, cls in enumerate(self.classes_)}
        return self

    @property
    def unknown_index(self) -> int:
        return len(self.classes_)

    def transform(self, values: Sequence[str]) -> np.ndarray:
        return np.array(
            [self._index.get(v, self.unknown_index) for v in values],
            dtype=np.int64,
        )

    def transform_one(self, value: str) -> int:
        return self._index.get(value, self.unknown_index)


@dataclass
class PreprocessingArtifacts:
    encoders: Dict[str, SafeLabelEncoder]
    scaler: MinMaxScaler
    feature_columns: List[str]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: List[str]
    encoders: Dict[str, SafeLabelEncoder]
    scaler: MinMaxScaler
    train_rows: int = field(init=False)
    test_rows: int = field(init=False)

    def __post_init__(self) -> None:
        self.train_rows = int(self.X_train.shape[0])
        self.test_rows = int(self.X_test.shape[0])


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw NSL-KDD-style CSV using the project's fixed column schema."""
    path = Path(path)
    df = pd.read_csv(path, names=RAW_COLUMNS)
    return df


def _binarize_label(series: pd.Series) -> np.ndarray:
    return series.apply(lambda x: 0 if x == "normal" else 1).to_numpy(dtype=np.int64)


def prepare_training_data(
    raw_path: str | Path,
    processed_dir: str | Path,
    preprocessing_artifacts_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    persist: bool = True,
) -> PreparedData:
    """
    Full reusable preprocessing pipeline for the classical branch:
    load -> drop difficulty -> binarize label -> split -> fit encoders on
    train only -> encode -> fit scaler on train only -> scale.

    Persists encoders, scaler and feature column order to
    `preprocessing_artifacts_dir`, and the resulting arrays to
    `processed_dir`, unless persist=False (used by tests against tmp dirs).
    """
    df = load_raw(raw_path)
    df = df.drop(columns=[DIFFICULTY_COLUMN])

    y = _binarize_label(df[LABEL_COLUMN])
    X = df.drop(columns=[LABEL_COLUMN]).reset_index(drop=True)
    feature_columns = list(X.columns)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    encoders: Dict[str, SafeLabelEncoder] = {}
    X_train_enc = X_train_raw.copy()
    X_test_enc = X_test_raw.copy()
    for col in CATEGORICAL_COLUMNS:
        enc = SafeLabelEncoder().fit(X_train_raw[col].astype(str))
        encoders[col] = enc
        X_train_enc[col] = enc.transform(X_train_raw[col].astype(str))
        X_test_enc[col] = enc.transform(X_test_raw[col].astype(str))

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc[feature_columns].to_numpy(dtype=np.float64))
    X_test_scaled = scaler.transform(X_test_enc[feature_columns].to_numpy(dtype=np.float64))

    if persist:
        processed_dir = Path(processed_dir)
        preprocessing_artifacts_dir = Path(preprocessing_artifacts_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        preprocessing_artifacts_dir.mkdir(parents=True, exist_ok=True)

        np.save(processed_dir / "X_train.npy", X_train_scaled)
        np.save(processed_dir / "X_test.npy", X_test_scaled)
        np.save(processed_dir / "y_train.npy", y_train)
        np.save(processed_dir / "y_test.npy", y_test)

        joblib.dump(encoders, preprocessing_artifacts_dir / "label_encoders.joblib")
        joblib.dump(scaler, preprocessing_artifacts_dir / "scaler.joblib")
        with open(preprocessing_artifacts_dir / "feature_columns.json", "w") as f:
            json.dump(feature_columns, f, indent=2)

    return PreparedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        encoders=encoders,
        scaler=scaler,
    )


def load_preprocessing_artifacts(preprocessing_artifacts_dir: str | Path) -> PreprocessingArtifacts:
    """Load persisted encoders/scaler/feature order for inference-time use."""
    d = Path(preprocessing_artifacts_dir)
    encoders = joblib.load(d / "label_encoders.joblib")
    scaler = joblib.load(d / "scaler.joblib")
    with open(d / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return PreprocessingArtifacts(encoders=encoders, scaler=scaler, feature_columns=feature_columns)


def transform_sample(
    sample: Mapping[str, object],
    artifacts: PreprocessingArtifacts,
) -> np.ndarray:
    """
    Transform a single raw traffic sample (dict-like, raw column names ->
    raw values, as they'd arrive from feature extraction) into a
    model-ready (1, n_features) scaled array, using persisted artifacts.

    Raises KeyError with a clear message if a required feature is missing.
    """
    missing = [c for c in artifacts.feature_columns if c not in sample]
    if missing:
        raise KeyError(
            f"Sample is missing required feature columns: {missing}. "
            f"Expected exactly: {artifacts.feature_columns}"
        )

    row = []
    for col in artifacts.feature_columns:
        value = sample[col]
        if col in CATEGORICAL_COLUMNS:
            value = artifacts.encoders[col].transform_one(str(value))
        row.append(value)

    arr = np.array(row, dtype=np.float64).reshape(1, -1)
    return artifacts.scaler.transform(arr)


def transform_batch(
    df: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> np.ndarray:
    """Vectorized equivalent of transform_sample for a DataFrame of raw rows."""
    missing = [c for c in artifacts.feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required feature columns: {missing}")

    out = df[artifacts.feature_columns].copy()
    for col in CATEGORICAL_COLUMNS:
        out[col] = artifacts.encoders[col].transform(out[col].astype(str))
    return artifacts.scaler.transform(out.to_numpy(dtype=np.float64))
