"""
src/defense/simulated_executor.py

The only DefenseExecutor implementation in this project. Every action
genuinely mutates SimulatedNetworkState -- there is no code path that
returns "succeeded=True" without the corresponding state fields actually
having changed. Fault injection (test-only, isolated on
SimulatedNetworkState) produces a realistic PARTIAL state change on
failure -- not "nothing happened" -- so rollback has real partial state
to restore from, matching Scenario B's requirement.

SAFETY: this executor performs no real I/O of any kind. It does not open
sockets, run subprocesses, touch the filesystem beyond the existing
logging infrastructure, or call any external system. Everything it does
is a dict mutation on an in-memory Python object.
"""

from __future__ import annotations

import copy

from src.defense import action_catalog as ac
from src.defense.executor import DefenseExecutor, ExecutionResult
from src.defense.simulated_state import SimulatedNetworkState, SourceState


class SimulatedDefenseExecutor(DefenseExecutor):
    def __init__(self, state: SimulatedNetworkState) -> None:
        self.state = state

    def execute(self, action_type: str, target: str) -> ExecutionResult:
        if not ac.is_known_action(action_type):
            raise ValueError(f"Unknown action type: '{action_type}'")

        before = self.state.snapshot(target)
        source = self.state.get_or_create(target)

        if self.state.should_fail(target):
            self._apply_partial(action_type, source)
            after = self.state.snapshot(target)
            return ExecutionResult(
                action_type=action_type, target=target, succeeded=False,
                before_state=before, after_state=after,
                error=f"simulated executor failure applying {action_type} to {target} (fault-injected)",
            )

        self._apply_full(action_type, source)
        after = self.state.snapshot(target)
        return ExecutionResult(action_type=action_type, target=target, succeeded=True, before_state=before, after_state=after)

    def rollback(self, action_type: str, target: str, pre_action_snapshot: SourceState) -> ExecutionResult:
        """
        Restores the COMPLETE pre-action state (every field), not a
        manual reversal of the one field the action nominally touched --
        this is what makes rollback correct even when an action had
        multi-field side effects (e.g. BLOCK also sets rate_limited).
        """
        before = self.state.snapshot(target)
        self.state.restore(target, pre_action_snapshot)
        after = self.state.snapshot(target)
        succeeded = after == pre_action_snapshot
        return ExecutionResult(
            action_type=f"ROLLBACK_{action_type}", target=target, succeeded=succeeded,
            before_state=before, after_state=after,
            error=None if succeeded else "rollback did not converge to the pre-action snapshot",
        )

    # ---- full (successful) effects, one per action type ---------------------

    def _apply_full(self, action_type: str, source: SourceState) -> None:
        if action_type == ac.MONITOR:
            source.monitored = True
        elif action_type == ac.INCREASE_MONITORING:
            source.monitored = True
            source.monitoring_level = "INCREASED"
        elif action_type == ac.RATE_LIMIT:
            source.rate_limited = True
        elif action_type == ac.ISOLATE_SIMULATED_SOURCE:
            source.isolated = True
            source.suspended_sessions += source.active_sessions
            source.active_sessions = 0
        elif action_type == ac.BLOCK_SIMULATED_SOURCE:
            source.blocked = True
            source.rate_limited = True  # blocking implies rate-limiting in this simulated policy
            source.active_sessions = 0
        elif action_type == ac.TERMINATE_SIMULATED_SESSION:
            source.terminated = True
            source.active_sessions = 0
        else:
            raise ValueError(f"No effect defined for action type: '{action_type}'")

    # ---- partial (fault-injected) effects -- some fields change, not all ----

    def _apply_partial(self, action_type: str, source: SourceState) -> None:
        if action_type == ac.MONITOR:
            pass  # nothing to partially apply -- MONITOR is single-field; failure means no change at all
        elif action_type == ac.INCREASE_MONITORING:
            source.monitored = True  # side effect landed, the actual level flip did not
        elif action_type == ac.RATE_LIMIT:
            pass  # single-field action -- failure means no change
        elif action_type == ac.ISOLATE_SIMULATED_SOURCE:
            # sessions got suspended (a real side effect occurred) but the
            # isolated flag itself never got set -- a realistic partial
            # failure, and exactly the kind of inconsistent state rollback
            # needs to fully clean up, not just flip `isolated` back.
            source.suspended_sessions += source.active_sessions
            source.active_sessions = 0
        elif action_type == ac.BLOCK_SIMULATED_SOURCE:
            # rate_limited got applied, but the actual block never landed
            source.rate_limited = True
        elif action_type == ac.TERMINATE_SIMULATED_SESSION:
            pass  # single-field action -- failure means no change
        else:
            raise ValueError(f"No partial-failure effect defined for action type: '{action_type}'")
