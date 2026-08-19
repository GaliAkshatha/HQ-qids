"""
src/defense/executor.py

Abstract interface for a defense executor. Kept separate from
SimulatedDefenseExecutor specifically so the interface is replaceable
later (per the architectural requirement) without touching callers --
but no other implementation exists in this project, and none should be
built without an explicit, separate safety review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.defense.simulated_state import SourceState


@dataclass
class ExecutionResult:
    action_type: str
    target: str
    succeeded: bool  # the executor's own claim -- verification checks this independently, never trusts it alone
    before_state: SourceState
    after_state: SourceState
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DefenseExecutor(ABC):
    @abstractmethod
    def execute(self, action_type: str, target: str) -> ExecutionResult:
        raise NotImplementedError

    @abstractmethod
    def rollback(self, action_type: str, target: str, pre_action_snapshot: SourceState) -> ExecutionResult:
        raise NotImplementedError
