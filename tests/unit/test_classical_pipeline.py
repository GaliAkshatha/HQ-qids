import numpy as np
import pandas as pd
import pytest

from src.preprocessing.classical_pipeline import (
    CATEGORICAL_COLUMNS,
    SafeLabelEncoder,
    load_preprocessing_artifacts,
    prepare_training_data,
    transform_batch,
    transform_sample,
)


def test_prepare_training_data_persists_and_reloads_consistently(tmp_path, sample_traffic_path):
    processed_dir = tmp_path / "Data" / "processed"
    preprocessing_dir = tmp_path / "artifacts" / "preprocessing"

    prepared = prepare_training_data(
        raw_path=sample_traffic_path,
        processed_dir=processed_dir,
        preprocessing_artifacts_dir=preprocessing_dir,
        test_size=0.25,
        random_state=42,
    )

    assert prepared.train_rows + prepared.test_rows == 24
    assert prepared.X_train.shape[1] == len(prepared.feature_columns) == 41
    # scaled into [0, 1]
    assert prepared.X_train.min() >= 0.0 - 1e-9
    assert prepared.X_train.max() <= 1.0 + 1e-9

    # persisted files exist
    assert (processed_dir / "X_train.npy").exists()
    assert (preprocessing_dir / "label_encoders.joblib").exists()
    assert (preprocessing_dir / "scaler.joblib").exists()
    assert (preprocessing_dir / "feature_columns.json").exists()

    # reloading from disk gives byte-identical transformation for a
    # fresh raw sample -- this is the property the whole reusable
    # interface depends on.
    artifacts = load_preprocessing_artifacts(preprocessing_dir)
    assert artifacts.feature_columns == prepared.feature_columns

    # build a raw sample dict directly from the pipeline's own loader to
    # avoid re-deriving the raw column schema in the test
    from src.preprocessing.classical_pipeline import load_raw

    df = load_raw(sample_traffic_path)
    sample = df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()

    via_reloaded_artifacts = transform_sample(sample, artifacts)
    via_inmemory_artifacts = transform_sample(
        sample,
        type(artifacts)(
            encoders=prepared.encoders,
            scaler=prepared.scaler,
            feature_columns=prepared.feature_columns,
        ),
    )
    np.testing.assert_allclose(via_reloaded_artifacts, via_inmemory_artifacts)


def test_unseen_category_maps_to_unknown_bucket_without_raising():
    enc = SafeLabelEncoder().fit(["tcp", "udp"])
    # "icmp" was never seen during fit
    result = enc.transform(["tcp", "icmp", "udp"])
    assert result[0] == enc._index["tcp"]
    assert result[2] == enc._index["udp"]
    assert result[1] == enc.unknown_index  # unseen -> dedicated bucket, no exception


def test_transform_sample_raises_clear_error_on_missing_feature(tmp_path, sample_traffic_path):
    preprocessing_dir = tmp_path / "artifacts" / "preprocessing"
    prepare_training_data(
        raw_path=sample_traffic_path,
        processed_dir=tmp_path / "Data" / "processed",
        preprocessing_artifacts_dir=preprocessing_dir,
        test_size=0.25,
        random_state=42,
    )
    artifacts = load_preprocessing_artifacts(preprocessing_dir)

    incomplete_sample = {"duration": 0, "protocol_type": "tcp"}  # missing most features
    with pytest.raises(KeyError):
        transform_sample(incomplete_sample, artifacts)


def test_transform_batch_matches_transform_sample_row_by_row(tmp_path, sample_traffic_path):
    preprocessing_dir = tmp_path / "artifacts" / "preprocessing"
    prepare_training_data(
        raw_path=sample_traffic_path,
        processed_dir=tmp_path / "Data" / "processed",
        preprocessing_artifacts_dir=preprocessing_dir,
        test_size=0.25,
        random_state=42,
    )
    artifacts = load_preprocessing_artifacts(preprocessing_dir)

    from src.preprocessing.classical_pipeline import load_raw

    df = load_raw(sample_traffic_path).drop(columns=["label", "difficulty"])

    batch_result = transform_batch(df, artifacts)
    for i in range(len(df)):
        row_result = transform_sample(df.iloc[i].to_dict(), artifacts)
        np.testing.assert_allclose(batch_result[i], row_result[0], atol=1e-9)
