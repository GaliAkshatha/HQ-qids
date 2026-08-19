import time

import numpy as np

from src.contracts import QuantumResult
from src.routing.worker import run_job


class StubVerifier:
    """Configurable stand-in for QSVMVerifier/VQCVerifier -- never touches
    real quantum code, so these tests are fast and deterministic."""

    model_name = "VQC"

    def __init__(self, behaviors):
        # list of callables, one per call to verify(); last one repeats if exhausted
        self.behaviors = behaviors
        self.call_count = 0

    def verify(self, scaled_features, sample_id):
        idx = min(self.call_count, len(self.behaviors) - 1)
        self.call_count += 1
        return self.behaviors[idx](sample_id)


def success_result(sample_id):
    return QuantumResult(
        sample_id=sample_id, quantum_model="VQC", status="success",
        quantum_prediction="normal", quantum_confidence=0.9,
        class_probabilities={"normal": 0.9, "attack": 0.1},
    )


def failed_result(sample_id):
    return QuantumResult(sample_id=sample_id, quantum_model="VQC", status="failed", error="simulated failure")


def slow_then_success(delay):
    def _verify(sample_id):
        time.sleep(delay)
        return success_result(sample_id)
    return _verify


def test_succeeds_on_first_attempt_no_retry():
    verifier = StubVerifier([success_result])
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=1.0, max_retries=2, backoff_seconds=0.01,
        sleep_fn=lambda s: None,
    )
    assert record.quantum_result.status == "success"
    assert record.attempts_made == 1
    assert record.fallback_reason is None


def test_retries_after_failure_then_succeeds():
    verifier = StubVerifier([failed_result, failed_result, success_result])
    sleep_calls = []
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=1.0, max_retries=2, backoff_seconds=0.01,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    assert record.quantum_result.status == "success"
    assert record.attempts_made == 3
    assert len(sleep_calls) == 2  # backoff before 2nd and 3rd attempts


def test_exhausts_retries_and_returns_terminal_failure():
    verifier = StubVerifier([failed_result])
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=1.0, max_retries=2, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    assert record.quantum_result.status == "failed"
    assert record.attempts_made == 3  # 1 initial + 2 retries
    assert record.fallback_reason == "retries_exhausted"


def test_no_retries_configured_means_single_attempt():
    verifier = StubVerifier([failed_result])
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=1.0, max_retries=0, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    assert record.attempts_made == 1


def test_slow_call_is_bounded_by_timeout_not_the_full_sleep_duration():
    """Proves the caller unblocks at the timeout boundary, not after the
    slow call actually finishes -- this is what 'must not block' means."""
    verifier = StubVerifier([slow_then_success(2.0)])
    t0 = time.perf_counter()
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=0.2, max_retries=0, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # nowhere near the 2.0s the stub actually sleeps
    assert record.quantum_result.status == "failed"
    assert "timeout" in record.quantum_result.error
    assert record.fallback_reason == "timeout"


def test_timeout_then_recovery_on_retry():
    verifier = StubVerifier([slow_then_success(2.0), success_result])
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=0.2, max_retries=1, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    assert record.quantum_result.status == "success"
    assert record.attempts_made == 2


def test_verifier_that_raises_directly_never_escapes_the_worker():
    """Distinct from the failed_result() case: this verifier's verify()
    method itself raises an exception, rather than catching internally
    and returning a failed QuantumResult -- proving the worker guards
    against a misbehaving/broken verifier, not just a well-behaved one."""

    class RaisingVerifier:
        model_name = "VQC"

        def verify(self, scaled_features, sample_id):
            raise RuntimeError("simulated broken verifier")

    record = run_job(
        RaisingVerifier(), np.zeros(41), "s1", timeout_seconds=1.0, max_retries=1, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    assert record.quantum_result.status == "failed"
    assert "RuntimeError" in record.quantum_result.error
    assert record.fallback_reason == "retries_exhausted"


def test_quantum_execution_time_is_measured_and_positive():
    verifier = StubVerifier([success_result])
    record = run_job(
        verifier, np.zeros(41), "s1", timeout_seconds=1.0, max_retries=0, backoff_seconds=0.001,
        sleep_fn=lambda s: None,
    )
    assert record.quantum_execution_time_ms >= 0
    assert record.total_job_time_ms >= record.quantum_execution_time_ms
