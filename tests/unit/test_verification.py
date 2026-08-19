from src.defense import action_catalog as ac
from src.defense import verification as ver
from src.defense.simulated_executor import SimulatedDefenseExecutor
from src.defense.simulated_state import SimulatedNetworkState


def test_verification_succeeds_after_real_state_change():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    executor.execute(ac.RATE_LIMIT, "s1")

    result = ver.verify(state, ac.RATE_LIMIT, "s1")
    assert result.verified is True
    assert result.checked_fields["rate_limited"] is True


def test_verification_fails_when_state_was_never_changed():
    """Verification reads state directly -- it doesn't just trust that
    execute() was called."""
    state = SimulatedNetworkState()
    state.get_or_create("s1")  # target exists but nothing was ever applied

    result = ver.verify(state, ac.RATE_LIMIT, "s1")
    assert result.verified is False


def test_verification_fails_on_partial_fault_injected_state():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    state.inject_failure_for("s1")
    executor.execute(ac.ISOLATE_SIMULATED_SOURCE, "s1")

    result = ver.verify(state, ac.ISOLATE_SIMULATED_SOURCE, "s1")
    assert result.verified is False
    assert result.checked_fields["isolated"] is False


def test_verification_checks_multiple_fields_for_multifield_actions():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    executor.execute(ac.BLOCK_SIMULATED_SOURCE, "s1")

    result = ver.verify(state, ac.BLOCK_SIMULATED_SOURCE, "s1")
    assert result.verified is True
    assert set(result.checked_fields.keys()) == {"blocked", "rate_limited_as_side_effect", "sessions_dropped"}
    assert all(result.checked_fields.values())


def test_verify_rollback_confirms_complete_state_restoration():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    pre_snapshot = state.snapshot("s1")
    executor.execute(ac.ISOLATE_SIMULATED_SOURCE, "s1")
    executor.rollback(ac.ISOLATE_SIMULATED_SOURCE, "s1", pre_snapshot)

    result = ver.verify_rollback(state, "s1", pre_snapshot)
    assert result.verified is True
    assert all(result.checked_fields.values())


def test_verify_rollback_detects_incomplete_restoration():
    state = SimulatedNetworkState()
    pre_snapshot = state.snapshot("s1")
    source = state.get_or_create("s1")
    source.blocked = True  # manually leave state inconsistent, not matching snapshot

    result = ver.verify_rollback(state, "s1", pre_snapshot)
    assert result.verified is False
    assert result.checked_fields["blocked"] is False
