import pytest

from src.agents.application_observation import ApplicationObservation


def make_observation(**overrides):
    base = dict(
        timestamp="2026-01-01T00:00:00+00:00", session_id="s1", agent_id="a1", agent_type="normal",
        action_type="login", method="POST", endpoint="/api/auth/login", status_code=200,
        latency_ms=42.0, authenticated=True, validation_success=True,
        target_label="CONTROLLED_LOCAL_SUZUME", sequence_number=1,
    )
    base.update(overrides)
    return ApplicationObservation(**base)


def test_valid_observation():
    obs = make_observation()
    assert obs.status_code == 200


def test_rejects_invalid_target_label():
    with pytest.raises(ValueError):
        make_observation(target_label="NOT_A_REAL_LABEL")


def test_all_three_valid_target_labels_accepted():
    for label in ("REAL_SUZUME_INTERACTION", "CONTROLLED_LOCAL_SUZUME", "SYNTHETIC_STUB"):
        make_observation(target_label=label)


def test_to_dict_round_trips():
    obs = make_observation()
    d = obs.to_dict()
    assert d["session_id"] == "s1"
    assert d["action_type"] == "login"


def test_no_field_exists_on_the_dataclass_for_secrets():
    """Structural proof, not just convention: ApplicationObservation has
    no field named anything that could hold a password/token/cookie."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(ApplicationObservation)}
    forbidden_substrings = ("password", "token", "cookie", "secret", "jwt")
    for name in field_names:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"field '{name}' looks like it could hold a secret"
