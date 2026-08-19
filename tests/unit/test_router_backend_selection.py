import numpy as np

from src.contracts import DetectionResult, QuantumResult
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter


class CountingStubVerifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.call_count = 0

    def verify(self, scaled_features, sample_id):
        self.call_count += 1
        return QuantumResult(
            sample_id=sample_id, quantum_model=self.model_name, status="success",
            quantum_prediction="normal", quantum_confidence=0.8,
            class_probabilities={"normal": 0.8, "attack": 0.2},
        )


def make_policy(backend):
    return RoutingPolicyConfig(
        confidence_threshold=0.70, anomaly_threshold=0.70, disagreement_threshold=0.30,
        combination="any", quantum_backend=backend,
        circuit_breaker_failure_threshold=3, circuit_breaker_cooldown_seconds=30,
        timeout_seconds={"QSVM": 5.0, "VQC": 1.0}, max_retries=0, backoff_seconds=0.001,
        queue_max_workers=4,
    )


def make_detection():
    return DetectionResult(
        sample_id="s1", classical_prediction="normal", classical_confidence=0.3,
        class_probabilities={"normal": 0.3, "attack": 0.7}, anomaly_score=0.1, model_disagreement=0.05,
    )


def test_router_configured_for_vqc_never_touches_qsvm_object():
    vqc_stub = CountingStubVerifier("VQC")
    router = QuantumRouter(policy=make_policy("VQC"), verifier=vqc_stub, job_queue=QuantumJobQueue(max_workers=2))

    decision = router.route_and_wait("s1", np.zeros(41), make_detection(), timeout=5)

    assert decision.quantum_backend == "VQC"
    assert vqc_stub.call_count == 1
    # nothing QSVM-shaped exists anywhere in this router's object graph
    assert not hasattr(router, "qsvm_verifier")


def test_router_configured_for_qsvm_never_touches_vqc_object():
    qsvm_stub = CountingStubVerifier("QSVM")
    router = QuantumRouter(policy=make_policy("QSVM"), verifier=qsvm_stub, job_queue=QuantumJobQueue(max_workers=2))

    decision = router.route_and_wait("s1", np.zeros(41), make_detection(), timeout=5)

    assert decision.quantum_backend == "QSVM"
    assert qsvm_stub.call_count == 1
    assert not hasattr(router, "vqc_verifier")


def test_single_route_call_invokes_backend_exactly_once_not_both():
    vqc_stub = CountingStubVerifier("VQC")
    router = QuantumRouter(policy=make_policy("VQC"), verifier=vqc_stub, job_queue=QuantumJobQueue(max_workers=2))
    router.route_and_wait("s1", np.zeros(41), make_detection(), timeout=5)
    assert vqc_stub.call_count == 1  # not 2 -- confirms no automatic dual-backend invocation
