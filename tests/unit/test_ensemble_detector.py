import numpy as np
import pytest

from src.contracts import DetectionResult
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.preprocessing.classical_pipeline import load_raw


@pytest.fixture(scope="module")
def detector(trained_fixture_artifacts) -> EnsembleClassicalDetector:
    return EnsembleClassicalDetector.load(
        models_dir=trained_fixture_artifacts["models_dir"],
        preprocessing_dir=trained_fixture_artifacts["preprocessing_dir"],
    )


@pytest.fixture(scope="module")
def fixture_rows(sample_traffic_path):
    df = load_raw(sample_traffic_path)
    return df


def test_detect_returns_valid_detection_result(detector, fixture_rows):
    sample = fixture_rows.drop(columns=["label", "difficulty"]).iloc[0].to_dict()
    result = detector.detect(sample, sample_id="test-1")

    assert isinstance(result, DetectionResult)  # __post_init__ already validated ranges
    assert result.sample_id == "test-1"
    assert result.classical_prediction in ("normal", "attack")
    assert 0.0 <= result.classical_confidence <= 1.0
    assert 0.0 <= result.anomaly_score <= 1.0
    assert 0.0 <= result.model_disagreement <= 1.0
    assert set(result.class_probabilities.keys()) == {"normal", "attack"}
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6


def test_model_disagreement_matches_metadata_probabilities(detector, fixture_rows):
    sample = fixture_rows.drop(columns=["label", "difficulty"]).iloc[1].to_dict()
    result = detector.detect(sample, sample_id="test-2")

    rf_p = result.metadata["rf_probability_attack"]
    xgb_p = result.metadata["xgb_probability_attack"]
    assert abs(result.model_disagreement - abs(rf_p - xgb_p)) < 1e-9


def test_classical_confidence_is_averaged_prediction_probability(detector, fixture_rows):
    sample = fixture_rows.drop(columns=["label", "difficulty"]).iloc[2].to_dict()
    result = detector.detect(sample, sample_id="test-3")

    predicted_label = result.classical_prediction
    assert abs(result.classical_confidence - result.class_probabilities[predicted_label]) < 1e-9


def test_detect_batch_matches_detect_called_per_row(detector, fixture_rows):
    df = fixture_rows.drop(columns=["label", "difficulty"])
    ids = [f"row-{i}" for i in range(len(df))]

    batch_results = detector.detect_batch(df, sample_ids=ids)
    looped_results = [detector.detect(df.iloc[i].to_dict(), sample_id=ids[i]) for i in range(len(df))]

    assert len(batch_results) == len(looped_results) == len(df)
    for b, s in zip(batch_results, looped_results):
        assert b.sample_id == s.sample_id
        assert b.classical_prediction == s.classical_prediction
        assert abs(b.classical_confidence - s.classical_confidence) < 1e-9
        assert abs(b.anomaly_score - s.anomaly_score) < 1e-9


def test_detect_raises_on_incomplete_sample(detector):
    with pytest.raises(KeyError):
        detector.detect({"duration": 0, "protocol_type": "tcp"}, sample_id="bad")


def test_unseen_categorical_value_does_not_crash(detector, fixture_rows):
    sample = fixture_rows.drop(columns=["label", "difficulty"]).iloc[0].to_dict()
    sample["protocol_type"] = "totally_novel_protocol_never_seen"
    result = detector.detect(sample, sample_id="unseen-category")
    assert isinstance(result, DetectionResult)
