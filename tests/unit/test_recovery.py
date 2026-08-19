from src.defense import action_catalog as ac
from src.defense.recovery import attempt_recovery
from src.defense.simulated_executor import SimulatedDefenseExecutor
from src.defense.simulated_state import SimulatedNetworkState


def test_retry_succeeds_after_transient_failure_counts_as_self_healed():
    """SCENARIO A -- successful self-healing:
    initial state -> defense attempt fails -> recovery/retry executes ->
    intended protective state is actually reached -> independent
    verification confirms it -> recovery_status=SUCCESS -> self-healed."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)

    state.inject_failure_for("s1")
    pre_snapshot = state.snapshot("s1")
    first_attempt = executor.execute(ac.ISOLATE_SIMULATED_SOURCE, "s1")
    assert first_attempt.succeeded is False  # initial remediation genuinely failed

    # fault clears before the retry -- simulating a transient failure
    state.clear_injected_failure("s1")

    result = attempt_recovery(
        executor=executor, state=state, action_type=ac.ISOLATE_SIMULATED_SOURCE, target="s1",
        pre_action_snapshot=pre_snapshot, max_retries=2, rollback_on_failure=True, action_reversible=True,
    )

    assert result.retry_succeeded is True
    assert result.self_healed is True
    assert result.recovery_status == "SUCCESS"
    assert result.rollback_attempted is False  # never needed -- retry alone succeeded

    # independent confirmation the intended state was actually reached
    source = state.get("s1")
    assert source.isolated is True
    assert source.active_sessions == 0


def test_permanent_failure_exhausts_retries_then_rolls_back_completely():
    """SCENARIO B -- failed remediation with rollback:
    initial state -> defense partially changes state -> verification
    fails -> retry fails -> rollback executes -> state returns EXACTLY to
    the original pre-action state -> independent verification confirms
    rollback -> recovery_status=FAILED (NOT counted as self-healed) ->
    health reflects the intended action was never achieved."""
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)

    state.inject_failure_for("s1")  # permanent -- never cleared in this test
    pre_snapshot = state.snapshot("s1")
    first_attempt = executor.execute(ac.BLOCK_SIMULATED_SOURCE, "s1")
    assert first_attempt.succeeded is False
    # partial state exists after the failed initial attempt
    assert first_attempt.after_state.rate_limited is True  # partial side effect landed
    assert first_attempt.after_state.blocked is False       # but the actual block did not

    result = attempt_recovery(
        executor=executor, state=state, action_type=ac.BLOCK_SIMULATED_SOURCE, target="s1",
        pre_action_snapshot=pre_snapshot, max_retries=1, rollback_on_failure=True, action_reversible=True,
    )

    assert result.retry_succeeded is False
    assert result.self_healed is False  # explicitly NOT counted as self-healing
    assert result.rollback_attempted is True
    assert result.rollback_succeeded is True
    assert result.recovery_status == "FAILED"  # rollback succeeding does not flip this to SUCCESS

    # independent verification: state returned EXACTLY to the original --
    # every field, not just the one flag BLOCK nominally touches
    final_state = state.get("s1")
    assert final_state == pre_snapshot
    assert final_state.blocked == pre_snapshot.blocked
    assert final_state.rate_limited == pre_snapshot.rate_limited
    assert final_state.active_sessions == pre_snapshot.active_sessions


def test_non_reversible_action_permanent_failure_does_not_attempt_rollback():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    state.inject_failure_for("s1")
    pre_snapshot = state.snapshot("s1")
    executor.execute(ac.TERMINATE_SIMULATED_SESSION, "s1")

    result = attempt_recovery(
        executor=executor, state=state, action_type=ac.TERMINATE_SIMULATED_SESSION, target="s1",
        pre_action_snapshot=pre_snapshot, max_retries=1, rollback_on_failure=True, action_reversible=False,
    )

    assert result.rollback_attempted is False  # not reversible -- never even tried
    assert result.recovery_status == "FAILED"
    assert result.self_healed is False


def test_retry_count_respects_max_retries_configuration():
    state = SimulatedNetworkState()
    executor = SimulatedDefenseExecutor(state)
    state.inject_failure_for("s1")
    pre_snapshot = state.snapshot("s1")
    executor.execute(ac.RATE_LIMIT, "s1")

    result = attempt_recovery(
        executor=executor, state=state, action_type=ac.RATE_LIMIT, target="s1",
        pre_action_snapshot=pre_snapshot, max_retries=3, rollback_on_failure=False, action_reversible=True,
    )
    assert result.retry_count == 3  # exhausted exactly the configured number
    assert result.rollback_attempted is False  # rollback_on_failure=False honored
