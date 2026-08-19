"""
Stage C tests: dataset generation, session-level splitting/leakage
prevention, label integrity, classical detector, adaptation policy,
secret redaction.
"""

import threading
import time

import pytest

from src.agents.application_adaptation import DeterministicAdaptationPolicy
from src.agents.application_dataset import (
    DATASET_LABEL,
    FEATURE_NAMES,
    LabeledSample,
    generate_labeled_sessions,
    split_by_session,
    to_matrix,
)
from src.agents.application_detector import ApplicationSecurityDetector, evaluate
from src.agents.application_features import ApplicationFeatureVector
from src.agents.suzume_traffic_source import SuzumeTrafficSource
from tests.support.local_suzume_target import build_local_suzume_app


@pytest.fixture(scope="module")
def local_target_url():
    app = build_local_suzume_app()
    port = 5197

    def run():
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    threading.Thread(target=run, daemon=True).start()
    time.sleep(1.0)
    yield f"http://127.0.0.1:{port}"


def make_vec(session_id, **overrides):
    base = dict(
        session_id=session_id, window_size=5, request_rate=1.0, failed_auth_rate=0.0,
        validation_failure_rate=0.0, endpoint_switch_rate=0.0, repeated_resource_access_rate=0.0,
        invalid_resource_rate=0.0, auth_failure_burst=0.0, crud_anomaly_score=0.0,
        response_error_rate=0.0, latency_anomaly_score=0.0, session_action_entropy=0.0,
        target_label="CONTROLLED_LOCAL_SUZUME",
    )
    base.update(overrides)
    return ApplicationFeatureVector(**base)


def test_generate_labeled_sessions_produces_correct_counts(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    samples = generate_labeled_sessions(source, n_normal_sessions=3, n_adversarial_sessions=4, seed=1)
    assert len(samples) == 7
    normal = [s for s in samples if s.agent_type == "normal"]
    adversarial = [s for s in samples if s.agent_type == "adversarial"]
    assert len(normal) == 3
    assert len(adversarial) == 4


def test_dataset_label_is_agent_generated_not_historical(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    samples = generate_labeled_sessions(source, n_normal_sessions=1, n_adversarial_sessions=1, seed=2)
    for s in samples:
        assert s.dataset_label == DATASET_LABEL == "AGENT_GENERATED_LABELED_DATA"


def test_labels_come_from_agent_type_not_a_detector(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    samples = generate_labeled_sessions(source, n_normal_sessions=2, n_adversarial_sessions=2, seed=3)
    for s in samples:
        if s.agent_type == "normal":
            assert s.label == "normal"
        else:
            assert s.label == "anomalous"


def test_split_by_session_produces_disjoint_session_sets():
    samples = [
        LabeledSample(scenario_id="x", session_id=f"sess-{i}", agent_type="normal",
                       feature_window=make_vec(f"sess-{i}"), label="normal")
        for i in range(20)
    ]
    train, val, test = split_by_session(samples, train_frac=0.6, val_frac=0.2, seed=0)
    train_ids = {s.session_id for s in train}
    val_ids = {s.session_id for s in val}
    test_ids = {s.session_id for s in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert len(train) + len(val) + len(test) == 20


def test_split_is_by_session_not_row_when_multiple_rows_share_a_session():
    samples = [
        LabeledSample(scenario_id="x", session_id="shared", agent_type="normal", feature_window=make_vec("shared"), label="normal"),
        LabeledSample(scenario_id="x", session_id="shared", agent_type="normal", feature_window=make_vec("shared"), label="normal"),
        LabeledSample(scenario_id="y", session_id="other", agent_type="adversarial", feature_window=make_vec("other"), label="anomalous"),
    ]
    train, val, test = split_by_session(samples, train_frac=0.6, val_frac=0.2, seed=0)
    all_splits = [{s.session_id for s in split} for split in (train, val, test)]
    shared_appearances = sum(1 for split_ids in all_splits if "shared" in split_ids)
    assert shared_appearances == 1


def test_to_matrix_shape_matches_feature_names():
    samples = [LabeledSample(scenario_id="x", session_id="s1", agent_type="normal", feature_window=make_vec("s1"), label="normal")]
    X, y = to_matrix(samples)
    assert len(X[0]) == len(FEATURE_NAMES) == 11
    assert y == [0]


def test_to_matrix_labels_normal_zero_anomalous_one():
    samples = [
        LabeledSample(scenario_id="x", session_id="s1", agent_type="normal", feature_window=make_vec("s1"), label="normal"),
        LabeledSample(scenario_id="y", session_id="s2", agent_type="adversarial", feature_window=make_vec("s2"), label="anomalous"),
    ]
    X, y = to_matrix(samples)
    assert y == [0, 1]


def test_application_security_detector_loads_and_predicts():
    detector = ApplicationSecurityDetector(model_name="random_forest")
    vec = make_vec("test-session", failed_auth_rate=1.0, auth_failure_burst=1.0)
    result = detector.detect(vec, sample_id="test-1")
    assert result.classical_prediction in ("normal", "attack")
    assert 0.0 <= result.classical_confidence <= 1.0
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6
    assert result.metadata["source"] == "APPLICATION_SECURITY"


def test_evaluate_reports_all_required_metrics():
    metrics = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix", "false_positive_rate", "false_negative_rate"):
        assert key in metrics


def test_adaptation_backs_off_after_escalation():
    policy = DeterministicAdaptationPolicy()
    record = policy.decide_next_scenario(
        allowed_scenario_ids=["repeated_failed_login", "malformed_payload_probe", "invalid_resource_probe"],
        previous_scenario_id="repeated_failed_login", escalated=True, risk_level="HIGH",
    )
    assert record.next_scenario_id == "malformed_payload_probe"
    assert "escalated" in record.adaptation_decision


def test_adaptation_repeats_scenario_without_escalation():
    policy = DeterministicAdaptationPolicy()
    record = policy.decide_next_scenario(
        allowed_scenario_ids=["repeated_failed_login", "malformed_payload_probe"],
        previous_scenario_id="repeated_failed_login", escalated=False, risk_level="LOW",
    )
    assert record.next_scenario_id == "repeated_failed_login"


def test_adaptation_with_single_allowed_scenario_has_no_alternative():
    policy = DeterministicAdaptationPolicy()
    record = policy.decide_next_scenario(
        allowed_scenario_ids=["repeated_failed_login"], previous_scenario_id="repeated_failed_login", escalated=True,
    )
    assert record.next_scenario_id == "repeated_failed_login"
    assert "no alternative" in record.adaptation_decision


def test_adaptation_record_is_fully_inspectable():
    policy = DeterministicAdaptationPolicy()
    record = policy.decide_next_scenario(
        allowed_scenario_ids=["a", "b"], previous_scenario_id="a", escalated=True, risk_level="CRITICAL",
    )
    assert record.previous_scenario_id == "a"
    assert record.escalated is True
    assert record.risk_level == "CRITICAL"
    assert isinstance(record.adaptation_decision, str) and len(record.adaptation_decision) > 0


def test_generated_dataset_never_contains_secret_material(local_target_url):
    source = SuzumeTrafficSource(local_target_url, target_label="CONTROLLED_LOCAL_SUZUME")
    samples = generate_labeled_sessions(source, n_normal_sessions=1, n_adversarial_sessions=1, seed=4)
    for s in samples:
        serialized = str(s.feature_window.to_dict())
        for forbidden in ("password", "AccessToken", "refreshToken", "Bearer "):
            assert forbidden not in serialized
