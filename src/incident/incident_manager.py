"""
src/incident/incident_manager.py

IncidentManager is a TRUE orchestration layer: it calls
EnsembleClassicalDetector, QuantumRouter, HybridPipeline, and
DefenseEngine exactly as built (Phases 1, 3, 4, 5) and does none of
their work itself. Its own job is: incident creation, correlation,
lifecycle transitions, event recording, escalation, idempotency, and
incident-level metrics.

Constructed with already-built dependencies (dependency injection) --
it does not know how to build a detector or a router, only how to call
one.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from src.contracts import IncidentEvent, IncidentSnapshot
from src.contracts.incident import (
    ASSESSING,
    DEFENSE_ACTION_EXECUTED,
    DEFENSE_ACTION_SELECTED,
    DEFENSE_VERIFICATION_FAILED,
    DETECTED,
    DETECTION_CREATED,
    ESCALATED,
    HYBRID_DECISION_CREATED,
    IDEMPOTENT_SKIP,
    INCIDENT_ESCALATED,
    INCIDENT_RESOLVED,
    MITIGATING,
    QUANTUM_ROUTING_REQUESTED,
    QUANTUM_VERIFICATION_COMPLETED,
    QUANTUM_VERIFICATION_FAILED,
    RECOVERY,
    RECOVERY_FAILED,
    RECOVERY_STARTED,
    RECOVERY_SUCCEEDED,
    RESOLVED,
    RISK_ASSESSED,
    ROLLBACK_EXECUTED,
    VERIFYING,
)
from src.incident.correlation import CorrelationStrategy, SampleIdCorrelation, build_tracker_from_events
from src.incident.escalation import EscalationPolicyConfig, evaluate_escalation
from src.incident.event_store import EventStore, InMemoryEventStore
from src.incident.incident_state import TransitionRequest, reconstruct_snapshot, validate_transition
from src.incident.metrics import IncidentMetrics
from src.observability.logging_config import get_logger, log_event
from src.preprocessing.classical_pipeline import transform_sample

logger = get_logger("incident_manager")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConcurrentProcessingError(RuntimeError):
    pass


class IncidentManager:
    def __init__(
        self,
        detector,
        router,
        hybrid_pipeline,
        defense_engine,
        event_store: Optional[EventStore] = None,
        correlation_strategy: Optional[CorrelationStrategy] = None,
        escalation_policy: Optional[EscalationPolicyConfig] = None,
        metrics: Optional[IncidentMetrics] = None,
    ) -> None:
        self.detector = detector
        self.router = router
        self.hybrid_pipeline = hybrid_pipeline
        self.defense_engine = defense_engine

        self.event_store = event_store or InMemoryEventStore()
        self.correlation_strategy = correlation_strategy or SampleIdCorrelation()
        self.escalation_policy = escalation_policy or EscalationPolicyConfig.load()
        self.metrics = metrics or IncidentMetrics()

        self._processing_lock = threading.Lock()
        self._in_flight: Set[str] = set()

        # Reconstructed from the event store, not assumed empty -- this is
        # what makes persistence meaningful rather than just logging.
        #
        # incident_id is the idempotency identity (Decision, Phase 8): a
        # redelivered incident_id must be detected and skipped. _snapshots
        # is therefore keyed by incident_id, NOT correlation_id -- a
        # correlation_id (session/entity identity) may legitimately map to
        # MANY distinct incident_ids, each processed independently.
        # _correlation_index tracks that one-to-many relationship;
        # _sample_id_index exists because process()'s own natural
        # idempotency key is sample_id (it always mints a fresh incident_id
        # itself, so it can never receive a duplicate incident_id the way
        # the distributed incident-worker can via message redelivery).
        self._snapshots: Dict[str, IncidentSnapshot] = {}
        self._correlation_index: Dict[str, list] = {}
        self._sample_id_index: Dict[str, str] = {}
        self.repeated_incident_tracker = build_tracker_from_events(self.event_store.read_all())
        self._rebuild_snapshots_from_store()

    # ---- reconstruction -----------------------------------------------------

    def _rebuild_snapshots_from_store(self) -> None:
        all_events = self.event_store.read_all()

        incident_order: list = []
        seen_incidents = set()
        incident_to_correlation: Dict[str, str] = {}
        incident_to_sample_id: Dict[str, str] = {}

        for event in all_events:
            if event.incident_id not in seen_incidents:
                seen_incidents.add(event.incident_id)
                incident_order.append(event.incident_id)
                incident_to_correlation[event.incident_id] = event.correlation_id
            if event.event_type == DETECTION_CREATED:
                sample_id = event.payload.get("sample_id")
                if sample_id:
                    incident_to_sample_id[event.incident_id] = sample_id

        for incident_id in incident_order:
            snapshot = reconstruct_snapshot(incident_id, all_events)
            if snapshot is None:
                continue
            self._snapshots[incident_id] = snapshot

            correlation_key = incident_to_correlation[incident_id]
            self._correlation_index.setdefault(correlation_key, [])
            if incident_id not in self._correlation_index[correlation_key]:
                self._correlation_index[correlation_key].append(incident_id)

            sample_id = incident_to_sample_id.get(incident_id)
            if sample_id and snapshot.is_terminal:
                self._sample_id_index[sample_id] = incident_id

    # ---- public read API ------------------------------------------------------

    def get_incident(self, incident_id: str) -> Optional[IncidentSnapshot]:
        return self._snapshots.get(incident_id)

    def get_incident_by_correlation(self, correlation_key: str) -> Optional[IncidentSnapshot]:
        """
        Returns the MOST RECENT incident for this correlation key. Under
        SampleIdCorrelation (Phase 1-7's default and still the default
        here), a correlation key maps to exactly one incident, so this
        is unchanged and backward-compatible for every existing caller.
        Under a strategy like Phase 8's AgentSessionCorrelation, where
        multiple incidents can legitimately share one key, use
        get_incidents_by_correlation() for the full history.
        """
        incident_ids = self._correlation_index.get(correlation_key)
        if not incident_ids:
            return None
        return self._snapshots.get(incident_ids[-1])

    def get_incidents_by_correlation(self, correlation_key: str) -> list:
        """Full, ordered history of every incident sharing this
        correlation key -- the correct view once multiple distinct
        incidents can share one key (Phase 8's session correlation)."""
        incident_ids = self._correlation_index.get(correlation_key, [])
        return [self._snapshots[i] for i in incident_ids if i in self._snapshots]

    def get_events(self, incident_id: str) -> list:
        return [e for e in self.event_store.read_all() if e.incident_id == incident_id]

    def metrics_snapshot(self) -> Dict[str, object]:
        return self.metrics.snapshot()

    # ---- event/transition helpers ----------------------------------------------

    def _emit(self, incident_id, correlation_key, event_type, previous_state, new_state, reason, payload=None) -> str:
        payload = payload or {}
        if new_state != previous_state:
            validate_transition(TransitionRequest(incident_id, previous_state, new_state, reason, event_type))
        event = IncidentEvent(
            event_id=str(uuid.uuid4()), correlation_id=correlation_key, incident_id=incident_id,
            event_type=event_type, previous_state=previous_state, new_state=new_state,
            timestamp=_now_iso(), reason=reason, payload=payload,
        )
        self.event_store.append(event)
        log_event(
            logger, 20, "incident event", incident_id=incident_id, correlation_id=correlation_key,
            event_type=event_type, previous_state=previous_state, new_state=new_state, reason=reason,
        )
        return new_state

    def append_idempotent_skip(self, existing: IncidentSnapshot, reason: str) -> None:
        """Public: also called directly by the distributed incident-worker
        (Phase 7) when it observes a terminal incident for a correlation
        key that received a new (non-duplicate) event_id -- a second,
        incident-level idempotency layer distinct from Phase 7's own
        event_id-level dedup in StreamWorker."""
        event = IncidentEvent(
            event_id=str(uuid.uuid4()), correlation_id=existing.correlation_id, incident_id=existing.incident_id,
            event_type=IDEMPOTENT_SKIP, previous_state=existing.current_state, new_state=existing.current_state,
            timestamp=_now_iso(), reason=reason, payload={},
        )
        self.event_store.append(event)
        log_event(logger, 20, "idempotent skip", incident_id=existing.incident_id, correlation_id=existing.correlation_id, reason=reason)
        rebuilt = reconstruct_snapshot(existing.incident_id, self.event_store.read_all(existing.correlation_id))
        if rebuilt is not None:
            self._snapshots[existing.incident_id] = rebuilt

    # ---- main entry point -----------------------------------------------------

    def process(self, sample_id: str, raw_sample: dict) -> IncidentSnapshot:
        """
        Runs one sample through the complete Detection -> Routing ->
        Hybrid -> Risk -> Defense chain, recording every step as an
        IncidentEvent and driving the incident through its lifecycle to
        a terminal state (RESOLVED or ESCALATED).

        Idempotent on sample_id: process() always mints a fresh
        incident_id itself, so "redelivery of the same incident_id" is
        not a concept that applies to this API -- its natural unit of
        work is sample_id. Calling process() again with a sample_id that
        already produced a terminal incident returns that incident
        unchanged, with an IDEMPOTENT_SKIP event recorded, regardless of
        whether other, different sample_ids share its correlation_key.
        A non-terminal in-flight call for the SAME sample_id in THIS
        process raises ConcurrentProcessingError rather than racing.
        """
        correlation_key = self.correlation_strategy.correlation_key(sample_id)

        with self._processing_lock:
            existing_incident_id = self._sample_id_index.get(sample_id)
            if existing_incident_id is not None:
                existing = self._snapshots.get(existing_incident_id)
                if existing is not None and existing.is_terminal:
                    self.append_idempotent_skip(existing, "sample_id already processed to a terminal incident -- not reprocessing")
                    return self._snapshots[existing_incident_id]
            if sample_id in self._in_flight:
                raise ConcurrentProcessingError(f"sample_id='{sample_id}' is already being processed")
            self._in_flight.add(sample_id)

        try:
            return self._run_full_chain(sample_id, raw_sample, correlation_key)
        finally:
            with self._processing_lock:
                self._in_flight.discard(sample_id)

    def _run_full_chain(self, sample_id: str, raw_sample: dict, correlation_key: str) -> IncidentSnapshot:
        incident_id = str(uuid.uuid4())
        created_at = _now_iso()

        detection_result = self.detector.detect(raw_sample, sample_id=sample_id)

        scaled_features = transform_sample(raw_sample, self.detector.preprocessing)[0]
        routing_decision = self.router.route_and_wait(sample_id, scaled_features, detection_result, timeout=10)

        hybrid_decision, risk_assessment = self.hybrid_pipeline.process(detection_result, routing_decision)

        rollback_count_before = self.defense_engine.metrics.snapshot()["rollbacks"]
        defense_result = self.defense_engine.process(detection_result, hybrid_decision, risk_assessment)
        rollback_occurred = self.defense_engine.metrics.snapshot()["rollbacks"] > rollback_count_before

        return self.record_full_lifecycle(
            correlation_key=correlation_key, incident_id=incident_id, created_at=created_at,
            detection_result=detection_result, routing_decision=routing_decision,
            hybrid_decision=hybrid_decision, risk_assessment=risk_assessment, defense_result=defense_result,
            rollback_occurred=rollback_occurred,
        )

    def record_full_lifecycle(
        self,
        correlation_key: str,
        incident_id: str,
        created_at: str,
        detection_result,
        routing_decision,
        hybrid_decision,
        risk_assessment,
        defense_result,
        rollback_occurred: bool,
    ) -> IncidentSnapshot:
        """
        Owns the COMPLETE incident lifecycle from DETECTED through a
        terminal state, given the five evidence objects. Callable two
        ways: (1) internally by process(), which computes those five
        objects itself in a single process, or (2) directly by a
        distributed incident-worker (Phase 7) that received them via
        Redis Streams from upstream workers -- without that worker
        re-running detection/routing/hybrid/defense a second time, and
        without needing any state from a prior call in this process
        (the distributed case has no such prior call). This method
        contains zero ML/quantum/risk/defense logic of its own; it only
        drives the state machine and records events from evidence it's
        handed.

        rollback_occurred is passed in explicitly (not recomputed here)
        because in the distributed case it must be computed by whichever
        component actually owns the relevant state -- the defense-
        worker's own DefenseEngine.metrics -- which this method has no
        way to independently observe.
        """
        self.metrics.record_incident_created()
        repeated_count = self.repeated_incident_tracker.record_incident(correlation_key)

        # ---- DETECTED ----
        state = self._emit(incident_id, correlation_key, DETECTION_CREATED, DETECTED, DETECTED,
                            "classical detection completed", detection_result.to_dict())

        # ---- DETECTED -> ASSESSING ----
        state = self._emit(incident_id, correlation_key, QUANTUM_ROUTING_REQUESTED, state, ASSESSING,
                            "evaluating routing policy", {})

        # ---- ASSESSING -> VERIFYING -> MITIGATING, or ASSESSING -> MITIGATING ----
        if routing_decision.should_invoke_quantum:
            state = self._emit(incident_id, correlation_key, QUANTUM_ROUTING_REQUESTED, state, VERIFYING,
                                "quantum verification required", {"reason_codes": routing_decision.reason_codes})
            quantum_failed = routing_decision.decision_status != "success"
            if quantum_failed:
                state = self._emit(incident_id, correlation_key, QUANTUM_VERIFICATION_FAILED, state, VERIFYING,
                                    f"quantum fallback: {routing_decision.fallback_reason}",
                                    {"fallback_reason": routing_decision.fallback_reason, "quantum_backend": routing_decision.quantum_backend})
            else:
                state = self._emit(incident_id, correlation_key, QUANTUM_VERIFICATION_COMPLETED, state, VERIFYING,
                                    "quantum verification succeeded", routing_decision.quantum_result.to_dict())
            self.metrics.record_quantum_verification(failed=quantum_failed)
            triggering = QUANTUM_VERIFICATION_FAILED if quantum_failed else QUANTUM_VERIFICATION_COMPLETED
            state = self._emit(incident_id, correlation_key, triggering, state, MITIGATING,
                                "proceeding to mitigation after quantum verification", {})
        else:
            state = self._emit(incident_id, correlation_key, QUANTUM_ROUTING_REQUESTED, state, MITIGATING,
                                "quantum verification not required by routing policy", {})

        # ---- hybrid decision + risk (recorded within MITIGATING; no
        #      dedicated lifecycle state exists for this per the approved
        #      transition table) ----
        state = self._emit(incident_id, correlation_key, HYBRID_DECISION_CREATED, state, MITIGATING,
                            "hybrid decision computed", hybrid_decision.to_dict())
        state = self._emit(incident_id, correlation_key, RISK_ASSESSED, state, MITIGATING,
                            "risk assessed", risk_assessment.to_dict())

        # ---- defense ----
        state = self._emit(incident_id, correlation_key, DEFENSE_ACTION_SELECTED, state, MITIGATING,
                            "defense action selected by policy", {"risk_level": risk_assessment.risk_level})
        state = self._emit(incident_id, correlation_key, DEFENSE_ACTION_EXECUTED, state, MITIGATING,
                            f"defense action_status={defense_result.action_status}", defense_result.to_dict())
        self.metrics.record_defense_outcome(failed=defense_result.action_status == "FAILED")

        # ---- MITIGATING -> RECOVERY, if the initial attempt needed recovery ----
        recovery_engaged = defense_result.recovery_status != "NOT_ATTEMPTED"
        if recovery_engaged:
            state = self._emit(incident_id, correlation_key, DEFENSE_VERIFICATION_FAILED, state, RECOVERY,
                                "initial defense verification failed -- entering recovery", {})
            state = self._emit(incident_id, correlation_key, RECOVERY_STARTED, state, RECOVERY,
                                "recovery/retry started", {})
            if rollback_occurred:
                state = self._emit(incident_id, correlation_key, ROLLBACK_EXECUTED, state, RECOVERY,
                                    "rollback executed to restore pre-action state", {})
            if defense_result.recovery_status == "SUCCESS":
                state = self._emit(incident_id, correlation_key, RECOVERY_SUCCEEDED, state, RECOVERY,
                                    "recovery independently verified successful", {})
            else:
                state = self._emit(incident_id, correlation_key, RECOVERY_FAILED, state, RECOVERY,
                                    "recovery failed even after retry/rollback", {})
            self.metrics.record_recovery_attempt(succeeded=defense_result.recovery_status == "SUCCESS", rollback_occurred=rollback_occurred)

        # ---- escalation decision (config-driven, from real Phase 4/5 evidence) ----
        should_escalate, escalation_reasons = evaluate_escalation(
            risk_assessment, hybrid_decision, defense_result, repeated_count, self.escalation_policy,
        )

        if should_escalate:
            state = self._emit(incident_id, correlation_key, INCIDENT_ESCALATED, state, ESCALATED,
                                "; ".join(escalation_reasons), {"escalation_reasons": escalation_reasons})
        else:
            state = self._emit(incident_id, correlation_key, INCIDENT_RESOLVED, state, RESOLVED,
                                "no escalation condition met", {})

        resolved_at = _now_iso()
        self.metrics.record_incident_terminal(escalated=should_escalate, created_at_iso=created_at, resolved_at_iso=resolved_at)

        snapshot = reconstruct_snapshot(incident_id, self.event_store.read_all(correlation_key))
        assert snapshot is not None

        self._snapshots[incident_id] = snapshot
        self._correlation_index.setdefault(correlation_key, [])
        if incident_id not in self._correlation_index[correlation_key]:
            self._correlation_index[correlation_key].append(incident_id)
        if snapshot.is_terminal:
            self._sample_id_index[detection_result.sample_id] = incident_id

        return snapshot
