"""
src/agents/templates.py

Turns a ScenarioDefinition into an NSL-KDD-shaped raw_sample dict, by
sampling a REAL row from KDDTrain+.txt matching the scenario's
source_label and applying bounded, logged perturbation to its numeric
fields. Categorical fields (protocol_type, service, flag) are NEVER
perturbed -- changing them wouldn't be "bounded perturbation of a real
exemplar," it would be fabricating a different exemplar.

IMPORTANT: this produces bounded perturbations of real, historical
NSL-KDD rows. It does NOT simulate real network or application traffic,
and must never be described as doing so.

No network, socket, or subprocess calls anywhere in this module -- it
only reads the already-present Data/raw/KDDTrain+.txt file and does
in-memory arithmetic.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.contracts import ScenarioDefinition
from src.preprocessing.classical_pipeline import CATEGORICAL_COLUMNS, RAW_COLUMNS, load_raw

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = REPO_ROOT / "Data" / "raw" / "KDDTrain+.txt"

FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c not in ("label", "difficulty")]
RATE_FIELDS = {c for c in FEATURE_COLUMNS if "rate" in c}  # real NSL-KDD [0,1]-bounded rate columns


class ExemplarBank:
    """
    Loads real KDDTrain+.txt once and indexes row indices by label, so
    sampling "a real row matching this scenario's source_label" is O(1)
    lookup + random choice, not a fresh file scan per sample.
    """

    def __init__(self, raw_path: str | Path = DEFAULT_RAW_PATH) -> None:
        self._df = load_raw(raw_path)
        self._by_label: Dict[str, List[int]] = {}
        for label in self._df["label"].unique():
            self._by_label[label] = self._df.index[self._df["label"] == label].tolist()

    def available_labels(self) -> List[str]:
        return list(self._by_label.keys())

    def sample_index(self, source_label: str, rng: random.Random) -> int:
        indices = self._by_label.get(source_label)
        if not indices:
            raise ValueError(
                f"No real exemplar rows found for label '{source_label}'. "
                f"Available labels: {sorted(self._by_label.keys())}"
            )
        return rng.choice(indices)

    def row_dict(self, index: int) -> Dict[str, Any]:
        row = self._df.loc[index]
        return {col: row[col] for col in FEATURE_COLUMNS}

    def row_label(self, index: int) -> str:
        return self._df.loc[index, "label"]


def apply_perturbation(
    raw_row: Dict[str, Any], magnitude: float, rng: random.Random
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Applies bounded multiplicative jitter to numeric, non-categorical
    fields only. Returns (perturbed_dict, perturbed_fields_log) where the
    log records exactly which fields changed and by how much, for full
    per-sample auditability.
    """
    if not (0.0 <= magnitude <= 1.0):
        raise ValueError(f"perturbation magnitude must be in [0,1]. Received: {magnitude}")

    perturbed = dict(raw_row)
    log: Dict[str, Dict[str, Any]] = {}

    if magnitude == 0.0:
        return perturbed, log

    for field_name, value in raw_row.items():
        if field_name in CATEGORICAL_COLUMNS:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue

        jitter = 1.0 + rng.uniform(-magnitude, magnitude)
        new_value = value * jitter

        if field_name in RATE_FIELDS:
            new_value = min(1.0, max(0.0, new_value))
        else:
            new_value = max(0.0, new_value)

        if isinstance(value, int):
            new_value = int(round(new_value))
        else:
            new_value = round(float(new_value), 4)

        if new_value != value:
            log[field_name] = {"before": value, "after": new_value}
        perturbed[field_name] = new_value

    return perturbed, log


def generate_sample(
    scenario: ScenarioDefinition,
    exemplar_bank: ExemplarBank,
    magnitude: float,
    rng: random.Random,
) -> Tuple[Dict[str, Any], int, str, Dict[str, Dict[str, Any]]]:
    """
    Returns (raw_sample, source_exemplar_index, source_exemplar_label,
    perturbed_fields_log) -- everything needed to build a
    GeneratedTrafficRecord alongside this sample.
    """
    index = exemplar_bank.sample_index(scenario.source_label, rng)
    source_label = exemplar_bank.row_label(index)
    base_row = exemplar_bank.row_dict(index)
    perturbed_row, perturbed_fields = apply_perturbation(base_row, magnitude, rng)
    return perturbed_row, index, source_label, perturbed_fields
