import pytest

from src.agents.agent_action import ACTION_TYPES, ApplicationAgentAction


def test_valid_action_type_accepted():
    action = ApplicationAgentAction(action_type="login", payload={"email": "a@b.com", "password": "x"})
    assert action.action_type == "login"


def test_unknown_action_type_rejected():
    with pytest.raises(ValueError):
        ApplicationAgentAction(action_type="not_a_real_action")


def test_every_documented_action_type_is_constructible():
    for action_type in ACTION_TYPES:
        ApplicationAgentAction(action_type=action_type)


def test_action_types_only_contain_stage_a_verified_routes():
    # Spot-check: these must exist (verified real Suzume routes);
    # a few plausible-but-unverified ones must NOT exist.
    assert "create_application" in ACTION_TYPES
    assert "list_applications" in ACTION_TYPES
    assert "delete_action_item" in ACTION_TYPES
    assert "admin_delete_user" not in ACTION_TYPES  # no admin role exists in Suzume
    assert "flood_endpoint" not in ACTION_TYPES


def test_default_payload_is_empty_dict():
    action = ApplicationAgentAction(action_type="get_dashboard_summary")
    assert action.payload == {}


def test_immutable():
    action = ApplicationAgentAction(action_type="login")
    with pytest.raises(Exception):
        action.action_type = "logout"
