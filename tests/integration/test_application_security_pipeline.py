import json

import pytest

from src.agents.application_detector import ApplicationSecurityDetector
from src.agents.application_features import ApplicationFeatureVector
from src.agents.application_pipeline import SOURCE_LABEL, ApplicationSecurityPipeline
from src.incident.event_store import InMemoryEventStore
from src.quantum.qsvm_verifier import QSVMVerifier
from src.routing.policy import RoutingPolicyConfig

DATASET_PATH = "reports/stage_c/dataset.json"
APP_QSVM_DIR = "artifacts/models/application/quantum/qsvm"
APP_PREPROCESSING_DIR = "artifacts/preprocessing/application"


@pytest.fixture(scope="module")
def real_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline():
    detector = ApplicationSecurityDetector(model_name="random_forest")
    verifier = QSVMVerifier.load(models_dir=APP_QSVM_DIR, preprocessing_dir=APP_PREPROCESSING_DIR)
    routing_policy = RoutingPolicyConfig.load().with_overrides(quantum_backend="QSVM")
    return ApplicationSecurityPipeline(
        detector=detector, quantum_verifier=verifier, event_store=InMemoryEventStore(), routing_policy=routing_policy,
    )


def make_vector(row):
    return ApplicationFeatureVector(session_id=row["session_id"], window_size=1, target_label="CONTROLLED_LOCAL_SUZUME", **row["features"])


def test_real_application_pipeline_produces_terminal_incidents(pipeline, real_dataset):
    for row in real_dataset["test"][:5]:
        vec = make_vector(row)
        result = pipeline.process(vec, correlation_key=row["session_id"])
        assert result.detection_source == SOURCE_LABEL == "APPLICATION_SECURITY"
        assert result.incident_snapshot.is_terminal
        assert result.incident_snapshot.current_state in ("RESOLVED", "ESCALATED")


def test_normal_labeled_sessions_do_not_escalate_via_real_pipeline(pipeline, real_dataset):
    normal_rows = [r for r in real_dataset["test"] if r["label"] == "normal"][:3]
    for row in normal_rows:
        vec = make_vector(row)
        result = pipeline.process(vec, correlation_key=row["session_id"] + "-normalcheck")
        assert result.incident_snapshot.escalated is False


def test_pipeline_reuses_existing_defense_engine_action_vocabulary(pipeline, real_dataset):
    from src.defense.action_catalog import ALL_ACTIONS

    row = real_dataset["test"][0]
    vec = make_vector(row)
    result = pipeline.process(vec, correlation_key=row["session_id"] + "-actioncheck")
    events = pipeline.incident_manager.get_events(result.incident_snapshot.incident_id)
    defense_events = [e for e in events if e.event_type == "DEFENSE_ACTION_EXECUTED"]
    assert len(defense_events) == 1
    action = defense_events[0].payload["action"]
    assert action in ALL_ACTIONS
