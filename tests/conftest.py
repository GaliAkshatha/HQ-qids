import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SAMPLE_TRAFFIC_PATH = FIXTURES_DIR / "sample_traffic.csv"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_traffic_path() -> Path:
    return SAMPLE_TRAFFIC_PATH


@pytest.fixture(scope="session")
def trained_fixture_artifacts(tmp_path_factory):
    """
    Trains RF + XGBoost + Isolation Forest on the small (24-row) real-data
    fixture, through the exact same run_training() code path production
    uses, into an isolated tmp directory -- so tests never touch the real
    artifacts/ directory and stay fast (small n_estimators).

    Session-scoped: trained once, reused by every test that needs a
    working detector.
    """
    from src.detection.train_classical import run_training

    base = tmp_path_factory.mktemp("phase1_fixture_artifacts")
    processed_dir = base / "Data" / "processed"
    models_dir = base / "artifacts" / "models" / "classical"
    preprocessing_dir = base / "artifacts" / "preprocessing"

    report = run_training(
        raw_path=SAMPLE_TRAFFIC_PATH,
        processed_dir=processed_dir,
        models_dir=models_dir,
        preprocessing_dir=preprocessing_dir,
        test_size=0.25,
        random_state=42,
        rf_params=dict(n_estimators=10, max_depth=4, random_state=42, n_jobs=-1),
        xgb_params=dict(n_estimators=10, max_depth=3, learning_rate=0.3, random_state=42, eval_metric="logloss"),
        if_params=dict(n_estimators=10, random_state=42),
    )

    return {
        "report": report,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "preprocessing_dir": preprocessing_dir,
    }


@pytest.fixture(scope="session")
def trained_quantum_fixture_artifacts(trained_fixture_artifacts, tmp_path_factory):
    """
    Trains tiny QSVM + VQC verifiers on the same fixture-scale Phase 1
    output (18 train / 6 test rows from the 24-row real fixture), through
    the exact run_qsvm_training() / run_vqc_training() code paths, into an
    isolated tmp directory.

    Deliberately small (reps=1, maxiter=5, 8/4 subsample) purely to keep
    the quantum simulator's real cost low in tests -- this proves plumbing
    correctness, not accuracy, same as Phase 1's fixture tests.
    """
    from src.quantum.train_qsvm import run_qsvm_training
    from src.quantum.train_vqc import run_vqc_training

    base = tmp_path_factory.mktemp("phase2_fixture_artifacts")
    preprocessing_dir = base / "artifacts" / "preprocessing"
    qsvm_models_dir = base / "artifacts" / "models" / "quantum" / "qsvm"
    vqc_models_dir = base / "artifacts" / "models" / "quantum" / "vqc"

    tiny_feature_map = {"type": "ZZFeatureMap", "num_qubits": 4, "reps": 1}
    tiny_ansatz = {"type": "RealAmplitudes", "num_qubits": 4, "reps": 1}
    tiny_optimizer = {"type": "COBYLA", "maxiter": 5}

    qsvm_report = run_qsvm_training(
        processed_dir=trained_fixture_artifacts["processed_dir"],
        models_dir=qsvm_models_dir,
        preprocessing_dir=preprocessing_dir,
        train_subsample_size=8,
        test_subsample_size=4,
        feature_map_config=tiny_feature_map,
        random_state=42,
    )

    vqc_report = run_vqc_training(
        processed_dir=trained_fixture_artifacts["processed_dir"],
        models_dir=vqc_models_dir,
        preprocessing_dir=preprocessing_dir,
        train_subsample_size=8,
        test_subsample_size=4,
        feature_map_config=tiny_feature_map,
        ansatz_config=tiny_ansatz,
        optimizer_config=tiny_optimizer,
        random_state=42,
    )

    return {
        "qsvm_report": qsvm_report,
        "vqc_report": vqc_report,
        "qsvm_models_dir": qsvm_models_dir,
        "vqc_models_dir": vqc_models_dir,
        "preprocessing_dir": preprocessing_dir,
        "processed_dir": trained_fixture_artifacts["processed_dir"],
    }
