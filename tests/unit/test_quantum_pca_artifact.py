import numpy as np

from src.quantum.pca_artifact import (
    fit_quantum_pca,
    get_or_fit_quantum_pca,
    load_quantum_pca,
    save_quantum_pca,
    transform_to_quantum_features,
)


def test_fit_persist_reload_gives_identical_transform(tmp_path):
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 1, size=(50, 41))

    pca = fit_quantum_pca(X_train, n_components=4, random_state=42)
    save_quantum_pca(pca, tmp_path)

    reloaded = load_quantum_pca(tmp_path)

    sample = X_train[0]
    original_out = transform_to_quantum_features(sample, pca)
    reloaded_out = transform_to_quantum_features(sample, reloaded)
    np.testing.assert_allclose(original_out, reloaded_out)


def test_transform_to_quantum_features_reduces_to_4_dims(tmp_path):
    rng = np.random.default_rng(1)
    X_train = rng.uniform(0, 1, size=(50, 41))
    pca = fit_quantum_pca(X_train, n_components=4)

    single = transform_to_quantum_features(X_train[0], pca)
    assert single.shape == (1, 4)

    batch = transform_to_quantum_features(X_train[:10], pca)
    assert batch.shape == (10, 4)


def test_get_or_fit_is_idempotent_across_calls(tmp_path):
    rng = np.random.default_rng(7)
    X_train = rng.uniform(0, 1, size=(50, 41))

    pca_1 = get_or_fit_quantum_pca(X_train, tmp_path, n_components=4, random_state=42)
    # second call with DIFFERENT data should still reuse the persisted artifact,
    # not refit -- this is the property the shared QSVM/VQC training scripts rely on
    different_data = rng.uniform(5, 10, size=(50, 41))
    pca_2 = get_or_fit_quantum_pca(different_data, tmp_path, n_components=4, random_state=42)

    np.testing.assert_allclose(pca_1.components_, pca_2.components_)
    np.testing.assert_allclose(pca_1.mean_, pca_2.mean_)


def test_load_raises_clear_error_when_missing(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_quantum_pca(tmp_path / "does_not_exist")
