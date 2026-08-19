from src.defense import action_catalog as ac
from src.defense.simulated_executor import SimulatedDefenseExecutor
from src.defense.simulated_state import SimulatedNetworkState


def test_monitor_sets_monitored_flag():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.MONITOR, "s1")
    assert result.succeeded is True
    assert result.after_state.monitored is True


def test_increase_monitoring_changes_level():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.INCREASE_MONITORING, "s1")
    assert result.after_state.monitoring_level == "INCREASED"


def test_rate_limit_sets_flag():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.RATE_LIMIT, "s1")
    assert result.after_state.rate_limited is True


def test_isolate_sets_flag_and_suspends_sessions():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.ISOLATE_SIMULATED_SOURCE, "s1")
    assert result.before_state.active_sessions == 1
    assert result.after_state.isolated is True
    assert result.after_state.active_sessions == 0
    assert result.after_state.suspended_sessions == 1


def test_block_sets_flag_and_has_multifield_side_effects():
    """BLOCK touches THREE fields (blocked, rate_limited, active_sessions)
    -- this is what makes the rollback test meaningful."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.BLOCK_SIMULATED_SOURCE, "s1")
    assert result.after_state.blocked is True
    assert result.after_state.rate_limited is True
    assert result.after_state.active_sessions == 0


def test_terminate_is_not_reversible():
    spec = ac.get_action_spec(ac.TERMINATE_SIMULATED_SESSION)
    assert spec.reversible is False
    assert spec.rollback_action is None


def test_state_is_genuinely_inspectable_before_and_after():
    """Proves the simulator isn't a canned success -- state actually differs."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    result = executor.execute(ac.RATE_LIMIT, "s1")
    assert result.before_state.rate_limited != result.after_state.rate_limited
    # independent read from the state object itself confirms it too
    assert state.get("s1").rate_limited is True


def test_fault_injection_produces_partial_not_total_failure():
    """Fault injection creates realistic partial state (some fields
    changed, not the flag that matters), not 'nothing happened'."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    state.inject_failure_for("s1")

    result = executor.execute(ac.ISOLATE_SIMULATED_SOURCE, "s1")
    assert result.succeeded is False
    assert result.error is not None
    # partial: sessions got suspended, but isolated flag never set
    assert result.after_state.isolated is False
    assert result.after_state.active_sessions == 0
    assert result.after_state.suspended_sessions == 1


def test_fault_injection_is_isolated_to_specific_target():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    state.inject_failure_for("bad-target")

    good_result = executor.execute(ac.RATE_LIMIT, "good-target")
    bad_result = executor.execute(ac.RATE_LIMIT, "bad-target")

    assert good_result.succeeded is True
    assert bad_result.succeeded is False


def test_rollback_restores_complete_multifield_state_not_one_flag():
    """The required test: rollback restores ALL touched fields, not just
    the primary boolean -- exercised on BLOCK, which touches three."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)

    pre_snapshot = state.snapshot("s1")
    exec_result = executor.execute(ac.BLOCK_SIMULATED_SOURCE, "s1")
    assert exec_result.after_state.blocked is True
    assert exec_result.after_state.rate_limited is True
    assert exec_result.after_state.active_sessions == 0

    rollback_result = executor.rollback(ac.BLOCK_SIMULATED_SOURCE, "s1", pre_snapshot)
    assert rollback_result.succeeded is True
    assert rollback_result.after_state.blocked == pre_snapshot.blocked
    assert rollback_result.after_state.rate_limited == pre_snapshot.rate_limited
    assert rollback_result.after_state.active_sessions == pre_snapshot.active_sessions
    assert rollback_result.after_state == pre_snapshot  # every field, exactly


def test_rollback_restores_pre_existing_state_not_default_false():
    """Nuance: if rate_limited was ALREADY True from a prior independent
    action before BLOCK ran, BLOCK's rollback must not incorrectly clear
    it to False -- it must restore the exact pre-BLOCK value (True)."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)

    executor.execute(ac.RATE_LIMIT, "s1")  # rate_limited=True, independently
    pre_block_snapshot = state.snapshot("s1")
    assert pre_block_snapshot.rate_limited is True

    executor.execute(ac.BLOCK_SIMULATED_SOURCE, "s1")
    rollback_result = executor.rollback(ac.BLOCK_SIMULATED_SOURCE, "s1", pre_block_snapshot)

    assert rollback_result.after_state.blocked is False
    assert rollback_result.after_state.rate_limited is True  # restored to pre-existing True, not cleared
