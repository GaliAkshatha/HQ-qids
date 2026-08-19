import random

import pytest

from src.agents.contracts import ScenarioDefinition
from src.agents.templates import ExemplarBank, apply_perturbation, generate_sample
from src.preprocessing.classical_pipeline import CATEGORICAL_COLUMNS, load_preprocessing_artifacts, transform_sample


@pytest.fixture(scope="module")
def exemplar_bank(repo_root):
    return ExemplarBank(repo_root / "Data" / "raw" / "KDDTrain+.txt")


@pytest.fixture(scope="module")
def preprocessing_artifacts(repo_root):
    return load_preprocessing_artifacts(repo_root / "artifacts" / "preprocessing")


def test_exemplar_bank_finds_real_rows_for_known_labels(exemplar_bank):
    rng = random.Random(0)
    idx = exemplar_bank.sample_index("neptune", rng)
    assert exemplar_bank.row_label(idx) == "neptune"
    row = exemplar_bank.row_dict(idx)
    assert len(row) == 41


def test_exemplar_bank_raises_for_unknown_label(exemplar_bank):
    rng = random.Random(0)
    with pytest.raises(ValueError):
        exemplar_bank.sample_index("not_a_real_label", rng)


def test_perturbation_never_touches_categorical_fields(exemplar_bank):
    rng = random.Random(1)
    idx = exemplar_bank.sample_index("neptune", rng)
    row = exemplar_bank.row_dict(idx)
    perturbed, log = apply_perturbation(row, 0.3, rng)
    for cat_field in CATEGORICAL_COLUMNS:
        assert perturbed[cat_field] == row[cat_field]
        assert cat_field not in log


def test_perturbation_zero_magnitude_produces_identical_sample(exemplar_bank):
    rng = random.Random(2)
    idx = exemplar_bank.sample_index("normal", rng)
    row = exemplar_bank.row_dict(idx)
    perturbed, log = apply_perturbation(row, 0.0, rng)
    assert perturbed == row
    assert log == {}


def test_perturbation_rejects_out_of_bounds_magnitude(exemplar_bank):
    rng = random.Random(3)
    idx = exemplar_bank.sample_index("normal", rng)
    row = exemplar_bank.row_dict(idx)
    with pytest.raises(ValueError):
        apply_perturbation(row, 1.5, rng)
    with pytest.raises(ValueError):
        apply_perturbation(row, -0.1, rng)


def test_perturbation_keeps_rate_fields_within_zero_one(exemplar_bank):
    rng = random.Random(4)
    for _ in range(20):
        idx = exemplar_bank.sample_index("neptune", rng)
        row = exemplar_bank.row_dict(idx)
        perturbed, _ = apply_perturbation(row, 0.3, rng)
        for field_name, value in perturbed.items():
            if "rate" in field_name:
                assert 0.0 <= value <= 1.0


def test_perturbation_keeps_values_non_negative(exemplar_bank):
    rng = random.Random(5)
    for _ in range(20):
        idx = exemplar_bank.sample_index("smurf", rng)
        row = exemplar_bank.row_dict(idx)
        perturbed, _ = apply_perturbation(row, 0.3, rng)
        for field_name, value in perturbed.items():
            if field_name in CATEGORICAL_COLUMNS:
                continue
            if isinstance(value, (int, float)):
                assert value >= 0.0


def test_generated_sample_is_valid_against_real_preprocessing(exemplar_bank, preprocessing_artifacts):
    rng = random.Random(6)
    scenario = ScenarioDefinition(name="neptune_flood", category="attack", source_label="neptune", description="d", default_perturbation=0.1)
    sample, idx, source_label, perturbed_fields = generate_sample(scenario, exemplar_bank, 0.1, rng)

    assert source_label == "neptune"
    vec = transform_sample(sample, preprocessing_artifacts)
    assert vec.shape == (1, 41)


def test_generated_sample_matches_scenario_source_label(exemplar_bank):
    rng = random.Random(7)
    scenario = ScenarioDefinition(name="normal_browsing", category="normal", source_label="normal", description="d", default_perturbation=0.1)
    for _ in range(10):
        _sample, _idx, source_label, _ = generate_sample(scenario, exemplar_bank, 0.1, rng)
        assert source_label == "normal"


def test_templates_module_has_no_network_socket_subprocess_capability():
    import src.agents.templates as mod
    from tests.unit._agent_safety_check import assert_no_io_capability
    assert_no_io_capability(mod.__file__)
