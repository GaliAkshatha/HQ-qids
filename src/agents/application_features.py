"""
src/agents/application_features.py

Defines an application-level security feature representation computed
ONLY from real, observable ApplicationObservation telemetry -- not the
NSL-KDD 41-feature schema, and not forced into it. See
docs/APPLICATION_SECURITY_MODEL_BOUNDARY.md for the full reasoning
behind keeping these separate.

Every feature below is documented with: definition, calculation,
observation window, range, why it's security-relevant, and whether it's
directly observed or derived. Features are computed over a WINDOW of
ApplicationObservation objects belonging to one session (or a bounded
recent slice of one), reflecting that application-security signal is
inherently about a *sequence* of requests, not a single one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from src.agents.application_observation import ApplicationObservation

_AUTH_ACTION_TYPES = {"register", "login", "refresh"}
_DESTRUCTIVE_ACTION_PREFIXES = ("update_", "delete_")


@dataclass
class ApplicationFeatureVector:
    """
    All fields are in [0, 1] except request_rate (requests/second,
    unbounded >= 0) and session_action_entropy (bits, unbounded >= 0) --
    both documented explicitly below.
    """

    session_id: str
    window_size: int  # number of observations this vector was computed from

    request_rate: float
    failed_auth_rate: float
    validation_failure_rate: float
    endpoint_switch_rate: float
    repeated_resource_access_rate: float
    invalid_resource_rate: float
    auth_failure_burst: float
    crud_anomaly_score: float
    response_error_rate: float
    latency_anomaly_score: float
    session_action_entropy: float

    target_label: str  # propagated from the observations -- REAL_SUZUME_INTERACTION | CONTROLLED_LOCAL_SUZUME | SYNTHETIC_STUB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_application_features(
    observations: List[ApplicationObservation],
    baseline_latency_ms: Optional[float] = None,
    baseline_latency_stddev_ms: Optional[float] = None,
) -> ApplicationFeatureVector:
    """
    Computes an ApplicationFeatureVector from a window of observations
    (typically one session's observations so far, or a bounded recent
    slice of it). baseline_latency_ms/stddev, if provided, should come
    from a separately-computed population baseline (e.g. normal-agent
    traffic) -- if omitted, latency_anomaly_score is reported as 0.0
    (no anomaly signal available), not fabricated.
    """
    if not observations:
        raise ValueError("compute_application_features requires at least one observation")

    ordered = sorted(observations, key=lambda o: o.sequence_number)
    n = len(ordered)
    session_id = ordered[0].session_id
    target_label = ordered[0].target_label

    # ---- request_rate: requests/second, DIRECTLY OBSERVED -------------------
    # Definition: count of requests in the window divided by the wall-clock
    # span between the first and last observation's timestamps.
    # Observation window: the full window passed in.
    # Range: [0, inf).
    # Security relevance: elevated request_rate is the most direct signal
    # of automated/scripted interaction vs. human pacing.
    first_ts = _parse_ts(ordered[0].timestamp)
    last_ts = _parse_ts(ordered[-1].timestamp)
    span_seconds = max((last_ts - first_ts).total_seconds(), 0.001)
    request_rate = n / span_seconds

    # ---- failed_auth_rate: DERIVED --------------------------------------------
    # Definition: fraction of auth-related actions (register/login/refresh)
    # in the window whose status_code indicates failure (>=400 or None).
    # Window: same as passed in. Range: [0, 1].
    # Security relevance: a high rate of failed authentication attempts is
    # a classic credential-guessing/brute-force signature -- kept strictly
    # observational here (the adversarial agent itself is bounded and never
    # performs real brute-forcing at scale; this feature would also fire on
    # a legitimate user repeatedly mistyping a password).
    auth_obs = [o for o in ordered if o.action_type in _AUTH_ACTION_TYPES]
    failed_auth_rate = (
        sum(1 for o in auth_obs if o.status_code is None or o.status_code >= 400) / len(auth_obs)
        if auth_obs else 0.0
    )

    # ---- validation_failure_rate: DIRECTLY OBSERVED ----------------------------
    # Definition: fraction of all observations with validation_success=False.
    # Window: full window. Range: [0, 1].
    # Security relevance: a high rate of schema-invalid submissions suggests
    # either a broken client or deliberate malformed-input probing.
    validation_failure_rate = sum(1 for o in ordered if not o.validation_success) / n

    # ---- endpoint_switch_rate: DERIVED (sequence-based) -------------------------
    # Definition: fraction of consecutive observation pairs where the
    # endpoint differs from the previous one.
    # Window: full window (requires >=2 observations to be meaningful;
    # 0.0 for a single observation).
    # Range: [0, 1].
    # Security relevance: legitimate workflows have a natural rhythm
    # (several calls to one resource area before moving on); rapid,
    # near-constant switching between unrelated endpoints is consistent
    # with automated endpoint enumeration/scanning behavior.
    if n >= 2:
        switches = sum(1 for i in range(1, n) if ordered[i].endpoint != ordered[i - 1].endpoint)
        endpoint_switch_rate = switches / (n - 1)
    else:
        endpoint_switch_rate = 0.0

    # ---- repeated_resource_access_rate: DERIVED ---------------------------------
    # Definition: fraction of observations whose endpoint has already
    # appeared earlier in the window (i.e. is not that endpoint's first
    # occurrence).
    # Window: full window. Range: [0, 1].
    # Security relevance: repeated access to the identical resource --
    # especially outside a natural create-then-view pattern -- is
    # consistent with either polling/scraping or repeated-probe behavior
    # against one target.
    seen_endpoints = set()
    repeated_count = 0
    for o in ordered:
        if o.endpoint in seen_endpoints:
            repeated_count += 1
        seen_endpoints.add(o.endpoint)
    repeated_resource_access_rate = repeated_count / n

    # ---- invalid_resource_rate: DIRECTLY OBSERVED --------------------------------
    # Definition: fraction of observations with status_code == 404.
    # Window: full window. Range: [0, 1].
    # Security relevance: repeated 404s are consistent with ID-guessing /
    # enumeration of resource identifiers that don't belong to the caller.
    invalid_resource_rate = sum(1 for o in ordered if o.status_code == 404) / n

    # ---- auth_failure_burst: DERIVED (run-length) ---------------------------------
    # Definition: the longest CONSECUTIVE run of failed auth observations
    # (status_code >= 400 or None) among auth-action observations, in
    # sequence order, normalized by the total number of auth observations.
    # Window: full window. Range: [0, 1] (0 if no auth observations).
    # Security relevance: a burst of consecutive failures (as opposed to
    # failures scattered among successes) is a stronger brute-force signal
    # than the raw failed_auth_rate alone, since it captures temporal
    # clustering, not just overall proportion.
    if auth_obs:
        longest_run = 0
        current_run = 0
        for o in auth_obs:
            if o.status_code is None or o.status_code >= 400:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        auth_failure_burst = longest_run / len(auth_obs)
    else:
        auth_failure_burst = 0.0

    # ---- crud_anomaly_score: DERIVED ------------------------------------------------
    # Definition: fraction of observations whose action_type begins with
    # "update_" or "delete_" (i.e. mutating/destructive actions) out of
    # all observations.
    # Window: full window. Range: [0, 1].
    # Security relevance: legitimate workflows (per the Stage A audit) are
    # read/create-heavy; a disproportionate share of update/delete calls
    # relative to reads is an unusual CRUD balance worth flagging.
    destructive_count = sum(1 for o in ordered if o.action_type.startswith(_DESTRUCTIVE_ACTION_PREFIXES))
    crud_anomaly_score = destructive_count / n

    # ---- response_error_rate: DIRECTLY OBSERVED ---------------------------------------
    # Definition: fraction of observations with status_code >= 500, OR a
    # transport-level error (observation.error is not None).
    # Window: full window. Range: [0, 1].
    # Security relevance: server errors or transport failures induced by
    # the caller's own requests are consistent with malformed/edge-case
    # payload probing.
    response_error_rate = sum(1 for o in ordered if (o.status_code is not None and o.status_code >= 500) or o.error) / n

    # ---- latency_anomaly_score: DERIVED, requires an external baseline -----------------
    # Definition: |mean(window latency) - baseline_mean| / baseline_stddev
    # (a z-score-like deviation measure), clipped to [0, 1] via a simple
    # saturating transform. Returns 0.0 if no baseline was supplied --
    # NOT fabricated as a default "normal" value, just explicitly "no
    # anomaly signal available."
    # Window: full window. Range: [0, 1].
    # Security relevance: request latency outside the normal range for
    # this application can indicate either server-side strain from the
    # caller's own behavior or unusual payload complexity.
    if baseline_latency_ms is not None and baseline_latency_stddev_ms and baseline_latency_stddev_ms > 0:
        mean_latency = sum(o.latency_ms for o in ordered) / n
        z = abs(mean_latency - baseline_latency_ms) / baseline_latency_stddev_ms
        latency_anomaly_score = 1.0 - math.exp(-z / 3.0)  # saturating transform, z=3 -> ~0.63, asymptotes to 1.0
    else:
        latency_anomaly_score = 0.0

    # ---- session_action_entropy: DERIVED (information-theoretic) -----------------------
    # Definition: Shannon entropy (base 2, in bits) of the action_type
    # distribution within the window.
    # Window: full window. Range: [0, log2(number of distinct action types
    # observed)] -- unbounded in principle, bounded in practice by the
    # size of ACTION_TYPES (~45 known types, so a practical ceiling around
    # 5.5 bits for a window using every type equally).
    # Security relevance: LOW entropy (repeating one or two action types
    # over and over) is consistent with either a very narrow legitimate
    # task or repetitive probing; HIGH entropy (many different action
    # types in quick succession) is consistent with broad
    # enumeration/exploration behavior atypical of a focused legitimate
    # workflow.
    counts = Counter(o.action_type for o in ordered)
    session_action_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) + 0.0

    return ApplicationFeatureVector(
        session_id=session_id, window_size=n,
        request_rate=request_rate, failed_auth_rate=failed_auth_rate,
        validation_failure_rate=validation_failure_rate, endpoint_switch_rate=endpoint_switch_rate,
        repeated_resource_access_rate=repeated_resource_access_rate, invalid_resource_rate=invalid_resource_rate,
        auth_failure_burst=auth_failure_burst, crud_anomaly_score=crud_anomaly_score,
        response_error_rate=response_error_rate, latency_anomaly_score=latency_anomaly_score,
        session_action_entropy=session_action_entropy, target_label=target_label,
    )


def _parse_ts(ts: str):
    from datetime import datetime
    return datetime.fromisoformat(ts)
