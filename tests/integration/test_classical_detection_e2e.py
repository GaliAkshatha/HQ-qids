"""
tests/integration/test_classical_detection_e2e.py

Exercises the full production path: raw traffic row -> persisted
preprocessing artifacts -> persisted models -> DetectionResult, loading
everything fresh from disk (as a new process would), rather than reusing
in-memory objects from training. This is the test that would catch a bug
where training "works" in-process but the persisted artifacts don't
actually round-trip correctly.
"""

import json

from src.contracts import DetectionResult
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.detection.train_classical import run_training
from src.preprocessing.classical_pipeline import load_raw


def test_end_to_end_classical_detection_from_persisted_artifacts(tmp_path, sample_traffic_path):
    processed_dir = tmp_path / "Data" / "processed"
    models_dir = tmp_path / "artifacts" / "models" / "classical"
    preprocessing_dir = tmp_path / "artifacts" / "preprocessing"

    report = run_training(
        raw_path=sample_traffic_path,
        processed_dir=processed_dir,
        models_dir=models_dir,
        preprocessing_dir=preprocessing_dir,
        test_size=0.25,
        random_state=7,
        rf_params=dict(n_estimators=10, max_depth=4, random_state=7, n_jobs=-1),
        xgb_params=dict(n_estimators=10, max_depth=3, learning_rate=0.3, random_state=7, eval_metric="logloss"),
        if_params=dict(n_estimators=10, random_state=7),
    )

    # artifacts actually landed on disk
    assert (models_dir / "random_forest.joblib").exists()
    assert (models_dir / "xgboost.joblib").exists()
    assert (models_dir / "isolation_forest.joblib").exists()
    assert (models_dir / "metrics.json").exists()
    assert (preprocessing_dir / "label_encoders.joblib").exists()
    assert (preprocessing_dir / "scaler.joblib").exists()
    assert (preprocessing_dir / "feature_columns.json").exists()
    assert (preprocessing_dir / "isolation_forest_normalization.json").exists()

    with open(models_dir / "metrics.json") as f:
        persisted_metrics = json.load(f)
    assert persisted_metrics["metrics"]["accuracy"] == report.metrics["accuracy"]

    # simulate a brand new process: construct the detector purely from
    # what's on disk, no reference to the objects run_training() returned
    fresh_detector = EnsembleClassicalDetector.load(
        models_dir=models_dir,
        preprocessing_dir=preprocessing_dir,
    )

    raw_df = load_raw(sample_traffic_path)
    raw_samples = raw_df.drop(columns=["label", "difficulty"]).to_dict(orient="records")
    true_labels = raw_df["label"].apply(lambda x: "normal" if x == "normal" else "attack").tolist()

    produced_results = []
    for i, sample in enumerate(raw_samples):
        result = fresh_detector.detect(sample, sample_id=f"e2e-{i}")
        assert isinstance(result, DetectionResult)
        assert result.classical_prediction in ("normal", "attack")
        assert 0.0 <= result.anomaly_score <= 1.0
        produced_results.append(result)

    # not asserting accuracy here -- 24 rows / 10-tree toy models is not a
    # meaningful accuracy claim, this test proves plumbing correctness.
    assert len(produced_results) == len(raw_samples) == len(true_labels)

    # metrics.json contains everything the plan promised to report
    for key in (
        "accuracy", "precision", "recall", "f1",
        "false_positive_rate", "false_negative_rate", "confusion_matrix",
    ):
        assert key in persisted_metrics["metrics"]
    assert "random_forest" in persisted_metrics["training_time_seconds"]
    assert "xgboost" in persisted_metrics["training_time_seconds"]
    assert "isolation_forest" in persisted_metrics["training_time_seconds"]
    assert "per_sample_ms" in persisted_metrics["inference_time_seconds"]
