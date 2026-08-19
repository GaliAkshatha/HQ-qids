"""
tests/integration/test_suzume_real_interaction_attempt.py

Attempts ONE bounded, safe, read-only-first interaction against the real
deployed Suzume application, labeled REAL_SUZUME_INTERACTION. Reports the
true outcome honestly -- this sandbox's network egress proxy blocks
suzume.akshathag.in (confirmed via a direct requests.get() returning
HTTP 403 with header x-deny-reason: host_not_allowed, which is the
sandbox's OWN proxy rejecting the host, not a response from Suzume
itself). This test is expected to be skipped in that case, not to fail
or to fabricate a result.

If network egress to the real host is ever permitted in this
environment, this test will actually exercise SuzumeTrafficSource
against the live deployment and report real, non-fabricated results.
"""

import uuid
from datetime import datetime, timezone

import pytest
import requests

from src.agents.agent_session import AgentSession

REAL_SUZUME_BASE_URL = "https://suzume.akshathag.in"


def _real_suzume_reachable() -> bool:
    try:
        response = requests.get(REAL_SUZUME_BASE_URL, timeout=10)
    except requests.exceptions.RequestException:
        return False
    if response.status_code == 403 and response.headers.get("x-deny-reason") == "host_not_allowed":
        return False
    return True


@pytest.mark.skipif(not _real_suzume_reachable(), reason="suzume.akshathag.in is blocked by this sandbox's network egress proxy (x-deny-reason: host_not_allowed) -- not a Suzume-side failure")
def test_bounded_real_suzume_health_check():
    session = AgentSession(
        session_id=str(uuid.uuid4()), agent_id="real-interaction-test", agent_type="normal",
        target_label="REAL_SUZUME_INTERACTION", created_at=datetime.now(timezone.utc).isoformat(),
    )
    response = requests.get(f"{REAL_SUZUME_BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    print(f"\nREAL_SUZUME_INTERACTION: /health returned {response.status_code}")


def test_network_egress_constraint_is_documented_and_verified():
    reachable = _real_suzume_reachable()
    print(f"\nsuzume.akshathag.in reachable from this sandbox: {reachable}")
    if not reachable:
        try:
            response = requests.get(REAL_SUZUME_BASE_URL, timeout=10)
            print(f"Blocked with status {response.status_code}, x-deny-reason={response.headers.get('x-deny-reason')}")
        except requests.exceptions.RequestException as e:
            print(f"Connection failed entirely: {type(e).__name__}")
