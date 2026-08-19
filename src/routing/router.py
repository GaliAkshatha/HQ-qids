"""
src/routing/router.py

Ties together policy evaluation, the circuit breaker, and the async job
queue/worker into RoutingDecision objects.

route() is genuinely non-blocking for quantum work: if a job is
submitted, it returns immediately with decision_status="pending",
quantum_attempted=False, quantum_result=None, and a real job_id -- never
a decision that pretends a result exists before the job finishes.

get_result(job_id) and route_and_wait() both resolve through the same
underlying queue/worker path (job_queue.get_result -> the same wrapped
job function submitted by route()) -- no separate/duplicated execution
logic for the synchronous case.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

from src.contracts import DetectionResult, RoutingDecision
from src.observability.logging_config import get_logger, log_event
from src.quantum.base import QuantumVerifier
from src.routing.circuit_breaker import QuantumCircuitBreaker
from src.routing.job_queue import QuantumJobQueue
from src.routing.metrics import RouterMetrics
from src.routing.policy import RoutingPolicyConfig, evaluate_routing
from src.routing.worker import JobExecutionRecord, run_job

logger = get_logger("quantum_router")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QuantumRouter:
    def __init__(
        self,
        policy: RoutingPolicyConfig,
        verifier: QuantumVerifier,
        job_queue: Optional[QuantumJobQueue] = None,
        breaker: Optional[QuantumCircuitBreaker] = None,
        metrics: Optional[RouterMetrics] = None,
    ) -> None:
        """
        verifier: the single configured backend's QuantumVerifier instance
        (QSVMVerifier or VQCVerifier), matching policy.quantum_backend.
        The router does not select or construct backends itself -- the
        caller wires up whichever verifier the config points to. This
        keeps "backend selection is config-driven" honest: swapping
        backends means swapping the verifier passed in, not branching
        inside the router.
        """
        if verifier.model_name != policy.quantum_backend:
            raise ValueError(
                f"policy.quantum_backend='{policy.quantum_backend}' does not match "
                f"verifier.model_name='{verifier.model_name}'"
            )
        self.policy = policy
        self.verifier = verifier
        self.job_queue = job_queue or QuantumJobQueue(max_workers=policy.queue_max_workers)
        self.breaker = breaker or QuantumCircuitBreaker(
            failure_threshold=policy.circuit_breaker_failure_threshold,
            cooldown_seconds=policy.circuit_breaker_cooldown_seconds,
        )
        self.metrics = metrics or RouterMetrics()

        # job_id -> RoutingDecision snapshot at submission time, so
        # get_result()/route_and_wait() can build the updated decision
        # without the caller needing to keep the original around.
        self._pending_decisions: Dict[str, RoutingDecision] = {}

    def route(
        self,
        sample_id: str,
        scaled_features: np.ndarray,
        detection_result: DetectionResult,
    ) -> RoutingDecision:
        t0 = time.perf_counter()
        should_invoke, reason_codes, signal_values = evaluate_routing(detection_result, self.policy)
        backend = self.policy.quantum_backend

        if not should_invoke:
            routing_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.record_event(should_invoke=False, routing_latency_ms=routing_latency_ms)
            decision = RoutingDecision(
                sample_id=sample_id,
                decision_status="not_invoked",
                should_invoke_quantum=False,
                reason_codes=reason_codes,
                signal_values=signal_values,
                policy_thresholds=self.policy.threshold_snapshot(),
                quantum_backend=None,
                circuit_breaker_state=self.breaker.state(backend),
                quantum_available=self.breaker.is_available(backend),
                quantum_attempted=False,
                quantum_result=None,
                fallback_used=False,
                fallback_reason=None,
                job_id=None,
                routing_latency_ms=routing_latency_ms,
                timestamp=_now_iso(),
            )
            log_event(logger, 20, "Routing decision: not invoked", sample_id=sample_id, reason_codes=reason_codes)
            return decision

        circuit_state = self.breaker.state(backend)
        allowed = self.breaker.allow_request(backend)

        if not allowed:
            routing_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics.record_event(should_invoke=True, routing_latency_ms=routing_latency_ms)
            self.metrics.record_fallback("circuit_open")
            decision = RoutingDecision(
                sample_id=sample_id,
                decision_status="fallback",
                should_invoke_quantum=True,
                reason_codes=reason_codes,
                signal_values=signal_values,
                policy_thresholds=self.policy.threshold_snapshot(),
                quantum_backend=backend,
                circuit_breaker_state=circuit_state,
                quantum_available=False,
                quantum_attempted=False,
                quantum_result=None,
                fallback_used=True,
                fallback_reason="circuit_open",
                job_id=None,
                routing_latency_ms=routing_latency_ms,
                timestamp=_now_iso(),
            )
            log_event(logger, 30, "Routing decision: circuit open, fallback", sample_id=sample_id)
            return decision

        # allowed -- submit the job asynchronously and return immediately
        job_id = self.job_queue.submit(
            run_job,
            verifier=self.verifier,
            scaled_features=scaled_features,
            sample_id=sample_id,
            timeout_seconds=self.policy.timeout_for(backend),
            max_retries=self.policy.max_retries,
            backoff_seconds=self.policy.backoff_seconds,
        )
        self.metrics.record_quantum_invocation()

        routing_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.metrics.record_event(should_invoke=True, routing_latency_ms=routing_latency_ms)

        decision = RoutingDecision(
            sample_id=sample_id,
            decision_status="pending",
            should_invoke_quantum=True,
            reason_codes=reason_codes,
            signal_values=signal_values,
            policy_thresholds=self.policy.threshold_snapshot(),
            quantum_backend=backend,
            circuit_breaker_state=circuit_state,
            quantum_available=True,
            quantum_attempted=False,
            quantum_result=None,
            fallback_used=False,
            fallback_reason=None,
            job_id=job_id,
            routing_latency_ms=routing_latency_ms,
            timestamp=_now_iso(),
        )
        self._pending_decisions[job_id] = decision
        log_event(logger, 20, "Routing decision: quantum job submitted", sample_id=sample_id, job_id=job_id)
        return decision

    def get_result(self, job_id: str, timeout: Optional[float] = None):
        """
        Returns the resolved JobExecutionRecord, or None if still pending
        after `timeout` seconds. This is a thin pass-through to the job
        queue -- the breaker/metrics update already happened inside the
        job function itself (at real completion time), not here, so
        calling this multiple times or never at all doesn't affect
        correctness.
        """
        return self.job_queue.get_result(job_id, timeout=timeout)

    def route_and_wait(
        self,
        sample_id: str,
        scaled_features: np.ndarray,
        detection_result: DetectionResult,
        timeout: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Convenience for synchronous callers (tests, or any caller that
        prefers to wait). Built entirely on route() + get_result() --
        does not re-run verifier.verify() or duplicate the worker's
        timeout/retry logic.
        """
        initial = self.route(sample_id, scaled_features, detection_result)
        if initial.job_id is None:
            return initial  # not_invoked or fallback (circuit_open) -- nothing to wait for

        record: Optional[JobExecutionRecord] = self.get_result(initial.job_id, timeout=timeout)
        if record is None:
            # still pending after the caller's own wait budget -- return
            # the pending decision unchanged rather than fabricating an
            # outcome that hasn't happened
            return initial

        return self._resolve_decision(initial, record)

    def _resolve_decision(self, initial: RoutingDecision, record: JobExecutionRecord) -> RoutingDecision:
        backend = initial.quantum_backend
        result = record.quantum_result
        success = result.status == "success"

        self.metrics.record_quantum_outcome(
            backend=backend, success=success, execution_time_ms=record.quantum_execution_time_ms
        )
        if success:
            self.breaker.record_success(backend)
        else:
            self.breaker.record_failure(backend)
            self.metrics.record_fallback(record.fallback_reason or "retries_exhausted")

        queue_wait_ms = self.job_queue.get_queue_wait_time_ms(initial.job_id)

        resolved = RoutingDecision(
            sample_id=initial.sample_id,
            decision_status="success" if success else "fallback",
            should_invoke_quantum=True,
            reason_codes=initial.reason_codes,
            signal_values=initial.signal_values,
            policy_thresholds=initial.policy_thresholds,
            quantum_backend=backend,
            circuit_breaker_state=self.breaker.state(backend),
            quantum_available=self.breaker.is_available(backend),
            quantum_attempted=True,
            quantum_result=result,
            fallback_used=not success,
            fallback_reason=None if success else record.fallback_reason,
            job_id=initial.job_id,
            routing_latency_ms=initial.routing_latency_ms,
            queue_wait_time_ms=queue_wait_ms,
            quantum_execution_time_ms=record.quantum_execution_time_ms,
            total_quantum_job_time_ms=record.total_job_time_ms,
            metadata={"attempts_made": record.attempts_made},
            timestamp=_now_iso(),
        )
        log_event(
            logger, 20, "Routing decision resolved",
            sample_id=initial.sample_id, decision_status=resolved.decision_status,
            attempts_made=record.attempts_made,
        )
        return resolved

    def metrics_snapshot(self) -> Dict[str, object]:
        return self.metrics.snapshot()
