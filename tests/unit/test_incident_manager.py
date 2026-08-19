import dataclasses

import pytest

from src.contracts import DefenseResult, HybridDecision, QuantumResult, RiskAssessment, RoutingDecision
from src.defense.metrics import DefenseMetrics
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.incident.correlation import SampleIdCorrelation
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import InMemoryEventStore
from src.incident.incident_manager import ConcurrentProcessingError, IncidentManager
from src.preprocessing.classical_pipeline import load_raw


# ---- stubs for the downstream phases -- fast, deterministic, no real
# quantum/hybrid/defense computation, but real contract objects ---------

class StubRouter:
    def __init__(self, factory):
        self.factory = factory

    def route_and_wait(self, sample_id, scaled_features, detection_result, timeout=None):
        return self.factory(sample_id, detection_result)


class StubHybridPipeline:
    def __init__(self, factory):
        self.factory = factory

    def process(self, detection_result, routing_decision):
        return self.factory(detection_result, routing_decision)


class StubDefenseEngine:
    def __init__(self, factory):
        self.factory = factory
        self.metrics = DefenseMetrics()

    def process(self, detection_result, hybrid_decision, risk_assessment):
        result, rollback_occurred = self.factory(detection_result, hybrid_decision, risk_assessment)
        self.metrics.record(
            selected_action=result.action, action_status=result.action_status, health_status=result.health_status,
            initial_verification_failed=(result.recovery_status != "NOT_ATTEMPTED"),
            recovery_attempted=(result.recovery_status != "NOT_ATTEMPTED"),
            recovery_status=result.recovery_status, rollback_attempted=rollback_occurred,
        )
        return result


def make_routing_not_invoked(sample_id, detection_result):
    return RoutingDecision(sample_id=sample_id, decision_status="not_invoked", should_invoke_quantum=False)


def make_routing_success(sample_id, detection_result):
    qr = QuantumResult(
        sample_id=sample_id, quantum_model="VQC", status="success",
        quantum_prediction=detection_result.classical_prediction, quantum_confidence=0.9,
        class_probabilities={"normal": 0.1, "attack": 0.9},
    )
    return RoutingDecision(
        sample_id=sample_id, decision_status="success", should_invoke_quantum=True,
        quantum_backend="VQC", quantum_attempted=True, quantum_result=qr,
    )


def make_hybrid_low_risk(detection_result, routing_decision):
    hd = HybridDecision(
        sample_id=detection_result.sample_id, final_prediction="normal", final_confidence=0.9,
        quantum_used=routing_decision.should_invoke_quantum, quantum_model=routing_decision.quantum_backend,
        decision_status="normal", evidence={"anomaly_score": 0.1, "model_disagreement": 0.05},
    )
    ra = RiskAssessment(
        sample_id=detection_result.sample_id, risk_level="LOW", risk_score=0.1,
        threat_evidence_score=0.1, system_uncertainty_score=0.05,
    )
    return hd, ra


def make_hybrid_critical_risk(detection_result, routing_decision):
    hd = HybridDecision(
        sample_id=detection_result.sample_id, final_prediction="attack", final_confidence=0.97,
        quantum_used=routing_decision.should_invoke_quantum, quantum_model=routing_decision.quantum_backend,
        decision_status="confirmed", evidence={"anomaly_score": 0.9, "model_disagreement": 0.4},
    )
    ra = RiskAssessment(
        sample_id=detection_result.sample_id, risk_level="CRITICAL", risk_score=0.9,
        threat_evidence_score=0.95, system_uncertainty_score=0.1,
    )
    return hd, ra


def make_defense_success(detection_result, hybrid_decision, risk_assessment):
    result = DefenseResult(
        sample_id=detection_result.sample_id, severity=risk_assessment.risk_level, risk_score=risk_assessment.risk_score,
        action="MONITOR", action_status="EXECUTED", recovery_status="NOT_ATTEMPTED", health_status="HEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )
    return result, False


def make_defense_self_healed(detection_result, hybrid_decision, risk_assessment):
    result = DefenseResult(
        sample_id=detection_result.sample_id, severity=risk_assessment.risk_level, risk_score=risk_assessment.risk_score,
        action="ISOLATE_SIMULATED_SOURCE", action_status="EXECUTED", recovery_status="SUCCESS", health_status="HEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )
    return result, False  # self-healed via retry, no rollback needed


def make_defense_failed_with_rollback(detection_result, hybrid_decision, risk_assessment):
    result = DefenseResult(
        sample_id=detection_result.sample_id, severity=risk_assessment.risk_level, risk_score=risk_assessment.risk_score,
        action="BLOCK_SIMULATED_SOURCE", action_status="FAILED", recovery_status="FAILED", health_status="UNHEALTHY",
        rollback_available=True, timestamp="2026-01-01T00:00:00+00:00",
    )
    return result, True  # rollback occurred


@pytest.fixture()
def real_detector(trained_fixture_artifacts):
    return EnsembleClassicalDetector.load(
        models_dir=trained_fixture_artifacts["models_dir"], preprocessing_dir=trained_fixture_artifacts["preprocessing_dir"],
    )


@pytest.fixture()
def real_sample(sample_traffic_path):
    df = load_raw(sample_traffic_path)
    return df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()


def make_manager(real_detector, routing_factory, hybrid_factory, defense_factory, event_store=None):
    return IncidentManager(
        detector=real_detector,
        router=StubRouter(routing_factory),
        hybrid_pipeline=StubHybridPipeline(hybrid_factory),
        defense_engine=StubDefenseEngine(defense_factory),
        event_store=event_store or InMemoryEventStore(),
        correlation_strategy=SampleIdCorrelation(),
        escalation_policy=EscalationPolicyConfig(
            on_critical_risk=True, on_unresolved_conflict_at_high_or_above=True,
            on_defense_action_failed=True, on_recovery_failed=True,
            repeated_incident_enabled=True, repeated_incident_threshold=3,
        ),
    )


# ---- incident creation + straightforward resolution ------------------------

def test_low_risk_incident_resolves_without_quantum(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_success)
    snapshot = manager.process("s1", real_sample)

    assert snapshot.current_state == "RESOLVED"
    assert snapshot.escalated is False
    assert len(snapshot.event_ids) > 0

    events = manager.get_events(snapshot.incident_id)
    event_types = [e.event_type for e in events]
    assert "DETECTION_CREATED" in event_types
    assert "INCIDENT_RESOLVED" in event_types
    assert "QUANTUM_VERIFICATION_COMPLETED" not in event_types  # quantum never invoked


def test_incident_with_quantum_verification_resolves(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_success, make_hybrid_low_risk, make_defense_success)
    snapshot = manager.process("s2", real_sample)

    events = manager.get_events(snapshot.incident_id)
    event_types = [e.event_type for e in events]
    assert "QUANTUM_VERIFICATION_COMPLETED" in event_types
    # VERIFYING appears in the transition history
    assert any(e.new_state == "VERIFYING" for e in events)


def test_critical_risk_escalates(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_critical_risk, make_defense_success)
    snapshot = manager.process("s3", real_sample)

    assert snapshot.current_state == "ESCALATED"
    assert snapshot.escalated is True
    assert "CRITICAL_RISK" in snapshot.escalation_reasons


# ---- recovery flows ----------------------------------------------------------

def test_self_healed_defense_goes_through_recovery_to_resolved(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_self_healed)
    snapshot = manager.process("s4", real_sample)

    events = manager.get_events(snapshot.incident_id)
    event_types = [e.event_type for e in events]
    assert "RECOVERY_STARTED" in event_types
    assert "RECOVERY_SUCCEEDED" in event_types
    assert "ROLLBACK_EXECUTED" not in event_types
    assert any(e.new_state == "RECOVERY" for e in events)
    assert snapshot.current_state == "RESOLVED"


def test_failed_recovery_with_rollback_escalates(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_failed_with_rollback)
    snapshot = manager.process("s5", real_sample)

    events = manager.get_events(snapshot.incident_id)
    event_types = [e.event_type for e in events]
    assert "RECOVERY_STARTED" in event_types
    assert "ROLLBACK_EXECUTED" in event_types
    assert "RECOVERY_FAILED" in event_types
    assert snapshot.current_state == "ESCALATED"
    assert "RECOVERY_FAILED" in snapshot.escalation_reasons or "DEFENSE_ACTION_FAILED" in snapshot.escalation_reasons


# ---- idempotency ----------------------------------------------------------------

def test_reprocessing_terminal_incident_is_idempotent_no_downstream_calls(real_detector, real_sample):
    call_count = {"router": 0}

    def counting_routing_factory(sample_id, detection_result):
        call_count["router"] += 1
        return make_routing_not_invoked(sample_id, detection_result)

    manager = make_manager(real_detector, counting_routing_factory, make_hybrid_low_risk, make_defense_success)
    first = manager.process("s6", real_sample)
    assert call_count["router"] == 1

    second = manager.process("s6", real_sample)  # same sample_id again
    assert call_count["router"] == 1  # NOT incremented -- router never called a second time
    assert second.incident_id == first.incident_id
    assert second.current_state == first.current_state

    events = manager.get_events(first.incident_id)
    assert any(e.event_type == "IDEMPOTENT_SKIP" for e in events)


def test_repeated_idempotent_calls_do_not_duplicate_events(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_success)
    manager.process("s7", real_sample)
    first_count = len(manager.get_events(manager.get_incident_by_correlation("s7").incident_id))

    manager.process("s7", real_sample)
    manager.process("s7", real_sample)
    third_count = len(manager.get_events(manager.get_incident_by_correlation("s7").incident_id))

    # exactly 2 IDEMPOTENT_SKIP events added, not full re-processing worth of events
    assert third_count == first_count + 2


def test_duplicate_event_id_never_appended_twice_via_manager(real_detector, real_sample):
    manager = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_success)
    snapshot = manager.process("s8", real_sample)
    all_event_ids = [e.event_id for e in manager.get_events(snapshot.incident_id)]
    assert len(all_event_ids) == len(set(all_event_ids))  # all unique


# ---- restart / reconstruction --------------------------------------------------

def test_new_manager_instance_reconstructs_terminal_state_from_shared_store(real_detector, real_sample):
    shared_store = InMemoryEventStore()
    manager_1 = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_success, event_store=shared_store)
    original = manager_1.process("s9", real_sample)
    assert original.current_state == "RESOLVED"

    # brand new IncidentManager, same store -- simulates a fresh process
    manager_2 = make_manager(real_detector, make_routing_not_invoked, make_hybrid_low_risk, make_defense_success, event_store=shared_store)
    reconstructed = manager_2.get_incident_by_correlation("s9")

    assert reconstructed is not None
    assert reconstructed.current_state == "RESOLVED"
    assert reconstructed.incident_id == original.incident_id
    assert reconstructed.event_ids == original.event_ids


def test_reconstructed_terminal_incident_prevents_reprocessing(real_detector, real_sample):
    call_count = {"router": 0}

    def counting_routing_factory(sample_id, detection_result):
        call_count["router"] += 1
        return make_routing_not_invoked(sample_id, detection_result)

    shared_store = InMemoryEventStore()
    manager_1 = make_manager(real_detector, counting_routing_factory, make_hybrid_low_risk, make_defense_success, event_store=shared_store)
    manager_1.process("s10", real_sample)
    assert call_count["router"] == 1

    manager_2 = make_manager(real_detector, counting_routing_factory, make_hybrid_low_risk, make_defense_success, event_store=shared_store)
    manager_2.process("s10", real_sample)  # a "new process" processing the same sample again
    assert call_count["router"] == 1  # still 1 -- manager_2 never called the router


# ---- repeated incidents / escalation via correlation ---------------------------

def test_repeated_incidents_for_same_correlation_key_eventually_escalate(real_detector, real_sample):
    """Tests the repeated-incident escalation rule by intentionally
    reusing a correlation key across DIFFERENT incidents -- exactly as
    instructed, since NSL-KDD's sample_id uniqueness means this can't
    happen organically."""
    shared_store = InMemoryEventStore()

    class RepeatKeyCorrelation(SampleIdCorrelation):
        def correlation_key(self, sample_id, **kwargs):
            return "shared-target"  # force every sample onto the same key

    manager = IncidentManager(
        detector=real_detector, router=StubRouter(make_routing_not_invoked),
        hybrid_pipeline=StubHybridPipeline(make_hybrid_low_risk), defense_engine=StubDefenseEngine(make_defense_success),
        event_store=shared_store, correlation_strategy=RepeatKeyCorrelation(),
        escalation_policy=EscalationPolicyConfig(
            on_critical_risk=True, on_unresolved_conflict_at_high_or_above=True,
            on_defense_action_failed=True, on_recovery_failed=True,
            repeated_incident_enabled=True, repeated_incident_threshold=3,
        ),
    )

    # Each incident must resolve to a DIFFERENT terminal snapshot under
    # the SAME correlation key -- only possible because each is a
    # logically distinct incident_id.  We bypass idempotency by manually
    # clearing the manager's cached snapshot between calls (representing
    # a case where a real future correlation strategy legitimately
    # creates a new incident for a recurring target rather than treating
    # it as a duplicate of a resolved one).
    results = []
    for i in range(3):
        manager._snapshots.pop("shared-target", None)
        snap = manager.process(f"sample-{i}", real_sample)
        results.append(snap)

    assert results[0].escalated is False
    assert results[1].escalated is False
    assert results[2].escalated is True
    assert "REPEATED_INCIDENT_THRESHOLD_EXCEEDED" in results[2].escalation_reasons


# ---- concurrency guard --------------------------------------------------------

def test_concurrent_processing_of_same_key_raises():
    """Directly exercises the in-flight guard without needing real
    threading -- simulates a second call arriving while the first is
    (hypothetically) still marked in-flight."""
    from src.incident.event_store import InMemoryEventStore as Store

    manager = IncidentManager(
        detector=None, router=None, hybrid_pipeline=None, defense_engine=None,
        event_store=Store(),
    )
    manager._in_flight.add("busy-key")
    with pytest.raises(ConcurrentProcessingError):
        manager.process("busy-key", {})
