import time

import numpy as np
import pytest

from src.contracts import DetectionResult, QuantumResult
from src.routing.circuit_breaker import OPEN, QuantumCircuitBreaker
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter


class StubVerifier:
    model_name = "VQC"

    def __init__(self, behaviors, delay=0.0):
        self.behaviors = behaviors
        self.call_count = 0
        self.delay = delay

    def verify(self, scaled_features, sample_id):
        if self.delay:
            time.sleep(self.delay)
        idx = min(self.call_count, len(self.behaviors) - 1)
        self.call_count += 1
        return self.behaviors[idx](sample_id)


def success_result(sample_id):
    return QuantumResult(
        sample_id=sample_id, quantum_model="VQC", status="success",
        quantum_prediction="attack", quantum_confidence=0.88,
        class_probabilities={"normal": 0.12, "attack": 0.88},
    )


def failed_result(sample_id):
    return QuantumResult(sample_id=sample_id, quantum_model="VQC", status="failed", error="stub failure")


def make_policy(**overrides):
    base = dict(
        confidence_threshold=0.70, anomaly_threshold=0.70, disagreement_threshold=0.30,
        combination="any", quantum_backend="VQC",
        circuit_breaker_failure_threshold=2, circuit_breaker_cooldown_seconds=30,
        timeout_seconds={"QSVM": 5.0, "VQC": 1.0}, max_retries=1, backoff_seconds=0.001,
        queue_max_workers=4,
    )
    base.update(overrides)
    return RoutingPolicyConfig(**base)


def make_detection(confidence=0.95, anomaly=0.1, disagreement=0.05, sample_id="s1"):
    return DetectionResult(
        sample_id=sample_id, classical_prediction="normal", classical_confidence=confidence,
        class_probabilities={"normal": confidence, "attack": 1 - confidence},
        anomaly_score=anomaly, model_disagreement=disagreement,
    )


def make_router(verifier, policy=None):
    policy = policy or make_policy()
    return QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))


def test_route_skips_quantum_when_policy_says_no():
    router = make_router(StubVerifier([success_result]))
    dr = make_detection(confidence=0.95, anomaly=0.1, disagreement=0.05)
    decision = router.route("s1", np.zeros(41), dr)

    assert decision.decision_status == "not_invoked"
    assert decision.should_invoke_quantum is False
    assert decision.quantum_attempted is False
    assert decision.quantum_result is None
    assert decision.job_id is None


def test_route_is_non_blocking_and_returns_pending_with_job_id():
    verifier = StubVerifier([success_result], delay=0.3)
    router = make_router(verifier)
    dr = make_detection(confidence=0.3)  # triggers LOW_CONFIDENCE

    t0 = time.perf_counter()
    decision = router.route("s1", np.zeros(41), dr)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.2  # returned well before the stub's 0.3s delay
    assert decision.decision_status == "pending"
    assert decision.should_invoke_quantum is True
    assert decision.quantum_attempted is False
    assert decision.quantum_result is None  # must not pretend a result exists yet
    assert decision.job_id is not None


def test_get_result_resolves_after_route_returns_pending():
    verifier = StubVerifier([success_result], delay=0.1)
    router = make_router(verifier)
    dr = make_detection(confidence=0.3)

    decision = router.route("s1", np.zeros(41), dr)
    record = router.get_result(decision.job_id, timeout=5)
    assert record is not None
    assert record.quantum_result.status == "success"


def test_route_and_wait_resolves_to_success_and_reuses_route_path():
    router = make_router(StubVerifier([success_result]))
    dr = make_detection(confidence=0.3)

    decision = router.route_and_wait("s1", np.zeros(41), dr, timeout=5)
    assert decision.decision_status == "success"
    assert decision.quantum_attempted is True
    assert decision.quantum_result.status == "success"
    assert decision.quantum_result.quantum_prediction == "attack"
    assert decision.fallback_used is False


def test_route_and_wait_records_fallback_after_retries_exhausted():
    router = make_router(StubVerifier([failed_result]))  # always fails
    dr = make_detection(confidence=0.3)

    decision = router.route_and_wait("s1", np.zeros(41), dr, timeout=5)
    assert decision.decision_status == "fallback"
    assert decision.quantum_attempted is True
    assert decision.fallback_used is True
    assert decision.fallback_reason == "retries_exhausted"
    assert decision.quantum_result.status == "failed"


def test_circuit_open_produces_immediate_fallback_without_job_id():
    breaker = QuantumCircuitBreaker(failure_threshold=1, cooldown_seconds=9999)
    breaker.record_failure("VQC")  # force OPEN before any routing happens
    assert breaker.state("VQC") == OPEN

    router = QuantumRouter(
        policy=make_policy(), verifier=StubVerifier([success_result]),
        job_queue=QuantumJobQueue(max_workers=4), breaker=breaker,
    )
    dr = make_detection(confidence=0.3)
    decision = router.route("s1", np.zeros(41), dr)

    assert decision.decision_status == "fallback"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "circuit_open"
    assert decision.job_id is None
    assert decision.quantum_available is False


def test_repeated_failures_actually_open_the_circuit_end_to_end():
    router = make_router(StubVerifier([failed_result]), policy=make_policy(circuit_breaker_failure_threshold=2))
    dr = make_detection(confidence=0.3)

    d1 = router.route_and_wait("s1", np.zeros(41), dr, timeout=5)
    assert d1.fallback_reason == "retries_exhausted"
    d2 = router.route_and_wait("s2", np.zeros(41), dr, timeout=5)
    assert d2.fallback_reason == "retries_exhausted"

    # circuit should now be OPEN -- third call falls back immediately, no job
    d3 = router.route("s3", np.zeros(41), dr)
    assert d3.fallback_reason == "circuit_open"
    assert d3.job_id is None


def test_backend_mismatch_between_policy_and_verifier_raises_at_construction():
    policy = make_policy(quantum_backend="QSVM")
    with pytest.raises(ValueError):
        QuantumRouter(policy=policy, verifier=StubVerifier([success_result]))  # verifier.model_name == "VQC"


def test_metrics_snapshot_reflects_routed_and_skipped_events():
    router = make_router(StubVerifier([success_result]))
    router.route("s1", np.zeros(41), make_detection(confidence=0.95))  # skip
    router.route_and_wait("s2", np.zeros(41), make_detection(confidence=0.3), timeout=5)  # invoke, success

    snap = router.metrics_snapshot()
    assert snap["total_events"] == 2
    assert snap["quantum_candidates"] == 1
    assert snap["skipped_events"] == 1
    assert snap["quantum_invocations"] == 1
    assert snap["quantum_successes"] == 1
