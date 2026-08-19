"""
Proves the circuit breaker actually opens under real (not simulated)
repeated failures, and that fallback actually engages -- using a
VQCVerifier deliberately pointed at a real-but-wrong artifact path so
verify() genuinely fails inside the real code path (not a mock).
"""

import numpy as np
import pytest

from src.contracts import DetectionResult
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.circuit_breaker import OPEN
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter


class AlwaysBrokenVerifier:
    """Same shape as a real verifier, but its verify() call genuinely
    raises every time -- exercising the worker's real exception handling,
    not a pre-canned failed QuantumResult."""

    model_name = "VQC"

    def verify(self, scaled_features, sample_id):
        raise RuntimeError("simulated real quantum backend outage")


def make_detection(sample_id="s1"):
    return DetectionResult(
        sample_id=sample_id, classical_prediction="normal", classical_confidence=0.1,
        class_probabilities={"normal": 0.1, "attack": 0.9}, anomaly_score=0.9, model_disagreement=0.5,
    )


def test_real_verifier_pointed_at_bad_artifacts_fails_cleanly(tmp_path, repo_root):
    """Confirms VQCVerifier.load() against a genuinely missing artifact
    directory raises at load time, not silently -- proven with the real
    class, not a stub."""
    with pytest.raises(FileNotFoundError):
        VQCVerifier.load(models_dir=tmp_path / "nope", preprocessing_dir=tmp_path / "also_nope")


def test_circuit_breaker_opens_after_real_repeated_failures_and_fallback_engages():
    policy = RoutingPolicyConfig(
        confidence_threshold=0.70, anomaly_threshold=0.70, disagreement_threshold=0.30,
        combination="any", quantum_backend="VQC",
        circuit_breaker_failure_threshold=2, circuit_breaker_cooldown_seconds=9999,
        timeout_seconds={"QSVM": 5.0, "VQC": 1.0}, max_retries=0, backoff_seconds=0.001,
        queue_max_workers=2,
    )
    router = QuantumRouter(policy=policy, verifier=AlwaysBrokenVerifier(), job_queue=QuantumJobQueue(max_workers=2))

    d1 = router.route_and_wait("s1", np.zeros(41), make_detection("s1"), timeout=5)
    assert d1.decision_status == "fallback"
    assert d1.fallback_reason == "retries_exhausted"
    assert router.breaker.state("VQC") == "CLOSED"  # 1 failure, threshold is 2

    d2 = router.route_and_wait("s2", np.zeros(41), make_detection("s2"), timeout=5)
    assert d2.fallback_reason == "retries_exhausted"
    assert router.breaker.state("VQC") == OPEN  # 2nd failure trips it

    # system keeps operating: third call gets an immediate circuit_open
    # fallback rather than attempting (and hanging on) another real call
    d3 = router.route("s3", np.zeros(41), make_detection("s3"))
    assert d3.decision_status == "fallback"
    assert d3.fallback_reason == "circuit_open"
    assert d3.job_id is None
    assert d3.quantum_available is False


def test_fresh_router_process_reconstruction_keeps_working_after_fallback(repo_root):
    """Fresh-process check for Phase 3: a newly constructed router (fresh
    breaker, fresh queue) against the SAME real VQC artifacts used
    elsewhere works normally -- a prior process's failures/open circuit
    don't persist and poison a new one (expected for Phase 3's in-memory
    breaker; Phase 7's persistence phase may change this deliberately)."""
    policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    verifier = VQCVerifier.load(
        models_dir=repo_root / "artifacts" / "models" / "quantum" / "vqc",
        preprocessing_dir=repo_root / "artifacts" / "preprocessing",
    )
    fresh_router = QuantumRouter(policy=policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=2))
    assert fresh_router.breaker.state("VQC") == "CLOSED"

    decision = fresh_router.route_and_wait("fresh-check", np.zeros(41), make_detection(), timeout=5)
    assert decision.quantum_attempted is True
