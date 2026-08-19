from src.agents.agent_session import AgentSession


def make_session(**overrides):
    base = dict(session_id="s1", agent_id="a1", agent_type="normal", target_label="CONTROLLED_LOCAL_SUZUME", created_at="2026-01-01T00:00:00+00:00")
    base.update(overrides)
    return AgentSession(**base)


def test_starts_unauthenticated():
    session = make_session()
    assert session.is_authenticated() is False


def test_set_authenticated():
    session = make_session()
    session.set_authenticated("token123", "user1", "a@b.com")
    assert session.is_authenticated() is True
    assert session.access_token == "token123"
    assert session.user_id == "user1"


def test_clear_authentication():
    session = make_session()
    session.set_authenticated("token123", "user1", "a@b.com")
    session.clear_authentication()
    assert session.is_authenticated() is False
    assert session.access_token is None


def test_sequence_numbers_increment_and_start_at_one():
    session = make_session()
    assert session.next_sequence() == 1
    assert session.next_sequence() == 2
    assert session.next_sequence() == 3


def test_last_response_data_starts_none():
    session = make_session()
    assert session.last_response_data is None
