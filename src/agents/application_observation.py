"""
src/agents/application_observation.py

The application-level telemetry object produced by every
ApplicationTrafficSource.execute_action() call. Deliberately has NO
field capable of holding a password, JWT, refresh token, or cookie --
there is no such field to accidentally populate.

target_label distinguishes REAL_SUZUME_INTERACTION from
CONTROLLED_LOCAL_SUZUME from SYNTHETIC_STUB -- required so metrics are
never silently blended across these categories.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

TARGET_LABELS = {"REAL_SUZUME_INTERACTION", "CONTROLLED_LOCAL_SUZUME", "SYNTHETIC_STUB"}


@dataclass
class ApplicationObservation:
    timestamp: str
    session_id: str
    agent_id: str
    agent_type: str
    action_type: str
    method: str
    endpoint: str
    status_code: Optional[int]
    latency_ms: float
    authenticated: bool
    validation_success: bool
    target_label: str
    sequence_number: int
    validation_errors: Optional[Dict[str, Any]] = None
    response_size: Optional[int] = None
    error: Optional[str] = None
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.target_label not in TARGET_LABELS:
            raise ValueError(f"target_label must be one of {TARGET_LABELS}. Received: {self.target_label}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
