import pytest

from src.contracts.incident import (
    ASSESSING, DETECTED, ESCALATED, MITIGATING, RECOVERY, RESOLVED, VERIFYING,
)
from src.incident.incident_state import InvalidTransitionError, TransitionRequest, validate_transition


def make_request(previous, new, reason="ok", event="DETECTION_CREATED"):
    return TransitionRequest(incident_id="inc-1", previous_state=previous, new_state=new, reason=reason, triggering_event=event)


@pytest.mark.parametrize("previous,new", [
    (DETECTED, ASSESSING),
    (ASSESSING, VERIFYING),
    (ASSESSING, MITIGATING),
    (VERIFYING, MITIGATING),
    (MITIGATING, RECOVERY),
    (MITIGATING, RESOLVED),
    (MITIGATING, ESCALATED),
    (RECOVERY, RESOLVED),
    (RECOVERY, ESCALATED),
])
def test_every_valid_transition_is_accepted(previous, new):
    validate_transition(make_request(previous, new))  # must not raise


@pytest.mark.parametrize("previous,new", [
    (DETECTED, MITIGATING),
    (DETECTED, RESOLVED),
    (DETECTED, VERIFYING),
    (ASSESSING, RESOLVED),
    (ASSESSING, ESCALATED),
    (ASSESSING, RECOVERY),
    (VERIFYING, RESOLVED),
    (VERIFYING, ESCALATED),
    (VERIFYING, ASSESSING),
    (VERIFYING, RECOVERY),
    (MITIGATING, ASSESSING),
    (MITIGATING, VERIFYING),
    (RECOVERY, MITIGATING),
    (RESOLVED, DETECTED),
    (RESOLVED, ASSESSING),
    (ESCALATED, RESOLVED),
    (ESCALATED, DETECTED),
])
def test_every_invalid_transition_raises(previous, new):
    with pytest.raises(InvalidTransitionError):
        validate_transition(make_request(previous, new))


def test_terminal_states_allow_no_further_transitions():
    for terminal in (RESOLVED, ESCALATED):
        with pytest.raises(InvalidTransitionError):
            validate_transition(make_request(terminal, ASSESSING))


def test_missing_reason_raises():
    with pytest.raises(InvalidTransitionError):
        validate_transition(make_request(DETECTED, ASSESSING, reason=""))


def test_missing_triggering_event_raises():
    with pytest.raises(InvalidTransitionError):
        validate_transition(make_request(DETECTED, ASSESSING, event=""))


def test_unknown_state_raises():
    with pytest.raises(InvalidTransitionError):
        validate_transition(make_request("NOT_A_STATE", ASSESSING))
