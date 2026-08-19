import pytest

from src.agents.application_observation import ApplicationObservation
from src.agents.application_features import compute_application_features


def make_obs(seq, action_type="list_applications", endpoint="/api/applications", status_code=200,
             validation_success=True, latency_ms=20.0, error=None, target_label="CONTROLLED_LOCAL_SUZUME"):
    return ApplicationObservation(
        timestamp=f"2026-01-01T00:00:{seq:02d}+00:00", session_id="s1", agent_id="a1", agent_type="normal",
        action_type=action_type, method="GET", endpoint=endpoint, status_code=status_code,
        latency_ms=latency_ms, authenticated=True, validation_success=validation_success,
        target_label=target_label, sequence_number=seq, error=error,
    )


def test_requires_at_least_one_observation():
    with pytest.raises(ValueError):
        compute_application_features([])


def test_request_rate_computed_from_real_span():
    obs = [make_obs(i) for i in range(1, 6)]  # 5 obs across a 4-second span (seq 1..5 as seconds)
    vec = compute_application_features(obs)
    assert vec.request_rate == pytest.approx(5 / 4, rel=0.01)


def test_failed_auth_rate_and_burst_from_repeated_failed_logins():
    obs = [make_obs(1, action_type="register", endpoint="/api/auth/register", status_code=201)]
    obs += [make_obs(i, action_type="login", endpoint="/api/auth/login", status_code=401) for i in range(2, 7)]
    vec = compute_application_features(obs)
    # register (success) + 5 failed logins = 5/6 auth observations failed
    assert vec.failed_auth_rate == pytest.approx(5 / 6)
    # all 5 failures are consecutive (the one success is first, not interleaved)
    assert vec.auth_failure_burst == pytest.approx(5 / 6)


def test_validation_failure_rate():
    obs = [make_obs(i, validation_success=(i % 2 == 0)) for i in range(1, 5)]
    vec = compute_application_features(obs)
    assert vec.validation_failure_rate == 0.5


def test_endpoint_switch_rate_high_when_always_different():
    obs = [make_obs(i, endpoint=f"/api/endpoint{i}") for i in range(1, 6)]
    vec = compute_application_features(obs)
    assert vec.endpoint_switch_rate == 1.0


def test_endpoint_switch_rate_zero_when_always_same():
    obs = [make_obs(i, endpoint="/api/applications") for i in range(1, 6)]
    vec = compute_application_features(obs)
    assert vec.endpoint_switch_rate == 0.0


def test_repeated_resource_access_rate():
    obs = [
        make_obs(1, endpoint="/api/applications/x"),
        make_obs(2, endpoint="/api/applications/y"),
        make_obs(3, endpoint="/api/applications/x"),  # repeat
    ]
    vec = compute_application_features(obs)
    assert vec.repeated_resource_access_rate == pytest.approx(1 / 3)


def test_invalid_resource_rate():
    obs = [make_obs(i, status_code=404 if i <= 3 else 200) for i in range(1, 6)]
    vec = compute_application_features(obs)
    assert vec.invalid_resource_rate == pytest.approx(3 / 5)


def test_crud_anomaly_score_high_for_destructive_heavy_session():
    obs = [make_obs(i, action_type="delete_application") for i in range(1, 4)]
    obs += [make_obs(4, action_type="list_applications")]
    vec = compute_application_features(obs)
    assert vec.crud_anomaly_score == pytest.approx(3 / 4)


def test_crud_anomaly_score_zero_for_read_only_session():
    obs = [make_obs(i, action_type="list_applications") for i in range(1, 4)]
    vec = compute_application_features(obs)
    assert vec.crud_anomaly_score == 0.0


def test_response_error_rate_counts_5xx_and_transport_errors():
    obs = [
        make_obs(1, status_code=500),
        make_obs(2, status_code=200, error="ConnectionError"),
        make_obs(3, status_code=200),
    ]
    vec = compute_application_features(obs)
    assert vec.response_error_rate == pytest.approx(2 / 3)


def test_latency_anomaly_score_zero_without_baseline():
    obs = [make_obs(i, latency_ms=5000.0) for i in range(1, 4)]  # very high latency, but no baseline given
    vec = compute_application_features(obs)
    assert vec.latency_anomaly_score == 0.0  # no fabricated anomaly signal


def test_latency_anomaly_score_nonzero_with_baseline_and_deviation():
    obs = [make_obs(i, latency_ms=1000.0) for i in range(1, 4)]
    vec = compute_application_features(obs, baseline_latency_ms=50.0, baseline_latency_stddev_ms=10.0)
    assert vec.latency_anomaly_score > 0.5


def test_session_action_entropy_zero_for_single_repeated_action():
    obs = [make_obs(i, action_type="login") for i in range(1, 6)]
    vec = compute_application_features(obs)
    assert vec.session_action_entropy == 0.0


def test_session_action_entropy_positive_for_varied_actions():
    obs = [
        make_obs(1, action_type="login"), make_obs(2, action_type="list_applications"),
        make_obs(3, action_type="get_dashboard_summary"), make_obs(4, action_type="get_analytics_overview"),
    ]
    vec = compute_application_features(obs)
    assert vec.session_action_entropy > 1.5  # 4 equally-likely distinct actions -> 2.0 bits


def test_target_label_propagated_from_observations():
    obs = [make_obs(1, target_label="REAL_SUZUME_INTERACTION")]
    vec = compute_application_features(obs)
    assert vec.target_label == "REAL_SUZUME_INTERACTION"


def test_window_size_matches_observation_count():
    obs = [make_obs(i) for i in range(1, 8)]
    vec = compute_application_features(obs)
    assert vec.window_size == 7
