import numpy as np

from src.detection.if_normalization import IsolationForestNormalization


def test_normalize_maps_lowest_raw_score_to_most_anomalous():
    norm = IsolationForestNormalization(score_min=-0.5, score_max=0.5)
    result = norm.normalize(np.array([-0.5]))
    assert result[0] == 1.0  # lowest raw score (most abnormal per sklearn) -> anomaly 1.0


def test_normalize_maps_highest_raw_score_to_least_anomalous():
    norm = IsolationForestNormalization(score_min=-0.5, score_max=0.5)
    result = norm.normalize(np.array([0.5]))
    assert result[0] == 0.0  # highest raw score (most normal) -> anomaly 0.0


def test_normalize_is_monotonic_decreasing_in_raw_score():
    norm = IsolationForestNormalization(score_min=-1.0, score_max=1.0)
    raw = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    normalized = norm.normalize(raw)
    assert np.all(np.diff(normalized) <= 0)  # increasing raw -> decreasing (or equal) anomaly


def test_normalize_clips_out_of_range_scores_seen_only_at_inference():
    norm = IsolationForestNormalization(score_min=-0.2, score_max=0.2)
    # more extreme than anything observed during training
    result = norm.normalize(np.array([-1.0, 1.0]))
    assert result[0] == 1.0
    assert result[1] == 0.0


def test_normalize_handles_degenerate_zero_span_without_dividing_by_zero():
    norm = IsolationForestNormalization(score_min=0.3, score_max=0.3)
    result = norm.normalize(np.array([0.3, 0.1, 0.9]))
    assert np.all(result == 0.5)  # neutral fallback, no NaN/inf
    assert np.all(np.isfinite(result))


def test_save_and_load_round_trip(tmp_path):
    norm = IsolationForestNormalization(score_min=-0.42, score_max=0.37)
    path = tmp_path / "if_norm.json"
    norm.save(path)
    reloaded = IsolationForestNormalization.load(path)
    assert reloaded.score_min == norm.score_min
    assert reloaded.score_max == norm.score_max
