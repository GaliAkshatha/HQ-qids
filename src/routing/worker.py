"""
src/routing/worker.py

Wraps a single QuantumVerifier.verify() call with a bounded timeout and a
retry policy. Runs the actual verify() call in its own thread (via a
ThreadPoolExecutor) so a hung/slow call can be bounded by
future.result(timeout=...) even though verify() itself has no internal
timeout support and Python threads can't be forcibly killed -- the CALLER
unblocks at the timeout boundary, which is what "must not block" means in
practice here.

Never raises. Every outcome -- success, per-attempt failure, or timeout --
becomes a QuantumResult, matching the same philosophy Phase 2's verifiers
already follow. sleep_fn is injectable so retry/backoff tests run in
milliseconds, not real wall-clock time.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from src.contracts import QuantumResult
from src.quantum.base import QuantumVerifier


@dataclass
class JobExecutionRecord:
    """Timing breakdown for one worker.run_job() call, independent of the
    RoutingDecision/job-queue-level queue_wait_time_ms, which the router
    adds separately."""

    quantum_result: QuantumResult
    attempts_made: int
    quantum_execution_time_ms: float  # sum of actual verify() call durations across all attempts
    total_job_time_ms: float  # attempts + backoff sleep, excludes queue wait
    fallback_reason: str | None  # "timeout" | "retries_exhausted" | None (success)


# Small internal executor dedicated to bounding individual verify() calls.
# Separate from the QuantumJobQueue's own executor (which runs whole jobs,
# i.e. run_job() itself) -- this one exists purely so a single attempt can
# be time-bounded via future.result(timeout=...).
_attempt_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="quantum-attempt")


def run_job(
    verifier: QuantumVerifier,
    scaled_features: np.ndarray,
    sample_id: str,
    timeout_seconds: float,
    max_retries: int,
    backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JobExecutionRecord:
    job_start = time.perf_counter()
    total_execution_ms = 0.0
    attempts = 0
    last_result: QuantumResult | None = None

    max_attempts = max_retries + 1
    while attempts < max_attempts:
        attempts += 1
        attempt_start = time.perf_counter()
        future = _attempt_executor.submit(verifier.verify, scaled_features, sample_id)
        try:
            result = future.result(timeout=timeout_seconds)
            attempt_ms = (time.perf_counter() - attempt_start) * 1000.0
            total_execution_ms += attempt_ms
        except FutureTimeoutError:
            attempt_ms = (time.perf_counter() - attempt_start) * 1000.0
            total_execution_ms += attempt_ms
            result = QuantumResult(
                sample_id=sample_id,
                quantum_model=verifier.model_name,
                status="failed",
                error=f"timeout after {timeout_seconds}s (attempt {attempts}/{max_attempts})",
                inference_time_ms=attempt_ms,
            )
        except Exception as e:  # noqa: BLE001 -- deliberate: a broken/misbehaving
            # verifier must never bring down the router. Phase 2's real
            # verifiers already catch their own exceptions internally and
            # return status="failed" instead of raising -- this branch
            # exists specifically for the case where that contract is
            # violated (bug, or a genuinely broken verifier), so even that
            # can't escape as an uncaught exception.
            attempt_ms = (time.perf_counter() - attempt_start) * 1000.0
            total_execution_ms += attempt_ms
            result = QuantumResult(
                sample_id=sample_id,
                quantum_model=verifier.model_name,
                status="failed",
                error=f"verifier raised {type(e).__name__}: {e}",
                inference_time_ms=attempt_ms,
            )

        last_result = result
        if result.status == "success":
            total_job_ms = (time.perf_counter() - job_start) * 1000.0
            return JobExecutionRecord(
                quantum_result=result,
                attempts_made=attempts,
                quantum_execution_time_ms=total_execution_ms,
                total_job_time_ms=total_job_ms,
                fallback_reason=None,
            )

        # failed attempt -- retry if attempts remain
        if attempts < max_attempts:
            sleep_fn(backoff_seconds)

    # exhausted all attempts without success
    total_job_ms = (time.perf_counter() - job_start) * 1000.0
    assert last_result is not None
    last_error = last_result.error or ""
    fallback_reason = "timeout" if last_error.startswith("timeout after") else "retries_exhausted"

    return JobExecutionRecord(
        quantum_result=last_result,
        attempts_made=attempts,
        quantum_execution_time_ms=total_execution_ms,
        total_job_time_ms=total_job_ms,
        fallback_reason=fallback_reason,
    )
