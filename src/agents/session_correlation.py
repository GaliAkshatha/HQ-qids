"""
src/agents/session_correlation.py

DISCREPANCY NOTE (resolved, not silently redesigned): the approved Phase
8 design described this class as implementing Phase 6's
CorrelationStrategy ABC directly (from src/incident/correlation.py).
Constraint #6 of this same approval explicitly forbids agent
implementations from importing incident/detection/quantum/routing/hybrid/
defense modules directly -- a real inconsistency between two parts of the
same approved design, not a repository mismatch. Resolved as follows:

AgentSessionCorrelation is a standalone, duck-typed class here, with NO
import from src/incident/ anywhere in this file. It implements the same
method shape (`correlation_key(self, sample_id, **kwargs) -> str`) that
Phase 6's CorrelationStrategy ABC expects, satisfying it structurally
(Python's duck typing) without formally subclassing it.

SECOND discrepancy, same resolution style: the approved implementation
order lists only ONE change to traffic_gateway.py (the constructor
parameter). Using CorrelationStrategy's kwargs-passing shape would have
required a second change -- ingest() would need to accept and forward
extra kwargs to reach correlation_key(sample_id, session_id=...). Instead,
this class parses session_id directly OUT OF sample_id, using a fixed,
documented delimiter -- so ingest()'s signature never needs to change.

Convention: agent-generated sample_ids must be formatted as
"<session_id>::<anything>", e.g. "sess-a1b2::turn-3::f9e1c2". Everything
before the first "::" is the session/correlation key; everything after is
free-form (turn index, a short random suffix, etc.) and only needs to be
unique within the session. src/agents/environment_engine.py is
responsible for constructing sample_ids this way.
"""

from __future__ import annotations

SESSION_DELIMITER = "::"


class AgentSessionCorrelation:
    def correlation_key(self, sample_id: str, **kwargs) -> str:
        if SESSION_DELIMITER not in sample_id:
            raise ValueError(
                f"AgentSessionCorrelation expects sample_id formatted as "
                f"'<session_id>{SESSION_DELIMITER}<anything>'. Received: '{sample_id}'"
            )
        session_id, _, _rest = sample_id.partition(SESSION_DELIMITER)
        if not session_id:
            raise ValueError(f"Empty session_id parsed from sample_id: '{sample_id}'")
        return session_id
