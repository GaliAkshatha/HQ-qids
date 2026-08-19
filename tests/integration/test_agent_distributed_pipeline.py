"""
The core Phase 8 proof: real agent-generated traffic (real NSL-KDD
exemplars + bounded perturbation) through the full real distributed
pipeline built in Phases 1-7 -- same real workers, same real Redis, same
real VQC artifacts, driven by the real, unmodified TrafficGateway.ingest()
seam. Also proves AgentSessionCorrelation lets repeated adversarial turns
organically trigger Phase 6's repeated_incident_threshold escalation rule
for the first time (previously only tested synthetically).
"""

import dataclasses
import uuid

import pytest
import redis

from src.agents.adversarial_agent import AdversarialAgent
from src.agents.environment_engine import EnvironmentEngine
from src.agents.feedback_listener import FeedbackListener
from src.agents.metrics import AgentRunMetrics
from src.agents.scenario_catalog import EnvironmentPolicy, ScenarioCatalog
from src.agents.session_correlation import AgentSessionCorrelation
from src.agents.templates import ExemplarBank
from src.runtime.config import RuntimePolicyConfig
from src.runtime.services import defense_worker, detection_worker, incident_worker, quantum_worker, risk_worker
from src.runtime.services.traffic_gateway import TrafficGateway


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_isolated_config() -> RuntimePolicyConfig:
    base = RuntimePolicyConfig.load()
    suffix = uuid.uuid4().hex[:8]
    streams = {k: f"{v}.agent-test-{suffix}" for k, v in base.streams.items()}
    groups = {k: f"{v}.agent-test-{suffix}" for k, v in base.consumer_groups.items()}
    return dataclasses.replace(base, streams=streams, consumer_groups=groups)


def drain(worker, n):
    for _ in range(n):
        worker.run_once(block_ms=3000)


def test_real_agent_traffic_through_full_real_distributed_pipeline(redis_client, repo_root, tmp_path):
    config = make_isolated_config()

    gateway = TrafficGateway(redis_client, config, correlation_strategy=AgentSessionCorrelation())
    det_worker = detection_worker.build_worker(config)
    q_worker = quantum_worker.build_worker(config)
    r_worker = risk_worker.build_worker(config)
    d_worker = defense_worker.build_worker(config)
    inc_worker = incident_worker.build_worker(config, event_store_path=tmp_path / "agent_events.jsonl")

    catalog = ScenarioCatalog.load()
    policy = EnvironmentPolicy.load()
    exemplar_bank = ExemplarBank(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=7)

    turns = 10
    turn_results = engine.run(turns=turns)

    drain(det_worker, turns)
    drain(q_worker, turns)
    drain(r_worker, turns)
    drain(d_worker, turns)
    drain(inc_worker, turns)

    listener = FeedbackListener(redis_client, config)
    metrics = AgentRunMetrics()

    for tr in turn_results:
        outcome = listener.wait_for_outcome(tr.pipeline_message.incident_id, timeout_ms=2000, poll_block_ms=100)
        assert outcome is not None, f"no outcome observed for incident {tr.pipeline_message.incident_id}"
        assert outcome.incident_current_state in ("RESOLVED", "ESCALATED")
        metrics.record(
            intended_label=tr.record.intended_label, scenario_name=tr.record.scenario_name,
            final_prediction=outcome.final_prediction, escalated=outcome.escalated,
            selected_action=outcome.selected_action, perturbation_magnitude=tr.record.perturbation_magnitude,
            agent_type=tr.record.agent_type, session_id=tr.record.session_id,
        )

    assert metrics.total_samples() == turns
    assert metrics.samples_with_observed_outcome() == turns
    print("\nReal agent-driven distributed run:")
    print(" confusion matrix:", metrics.confusion_matrix())
    print(" scenario distribution:", metrics.scenario_distribution())
    print(" escalation rate by scenario:", metrics.escalation_rate_by_scenario())
    print(" defense action distribution:", metrics.defense_action_distribution_by_scenario())


def test_repeated_adversarial_turns_organically_trigger_repeated_incident_escalation(redis_client, repo_root, tmp_path):
    """Experiment D: run several adversarial turns under ONE session
    identity and prove the real, existing repeated_incident_threshold
    escalation rule (Phase 6, config/incident_policy.json, threshold=3)
    fires organically -- not via a synthetic test double."""
    config = make_isolated_config()

    gateway = TrafficGateway(redis_client, config, correlation_strategy=AgentSessionCorrelation())
    det_worker = detection_worker.build_worker(config)
    q_worker = quantum_worker.build_worker(config)
    r_worker = risk_worker.build_worker(config)
    d_worker = defense_worker.build_worker(config)
    inc_worker = incident_worker.build_worker(config, event_store_path=tmp_path / "repeated_incident_events.jsonl")

    catalog = ScenarioCatalog.load()
    policy = EnvironmentPolicy.load()
    exemplar_bank = ExemplarBank(repo_root / "Data" / "raw" / "KDDTrain+.txt")

    adversarial = AdversarialAgent("repeat-attacker", policy.adversarial_allowed_scenarios)
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=11)

    session_id = "repeat-attack-session"
    turns = 5  # >= repeated_incident_threshold (3)
    turn_results = engine.run(turns=turns, agents=[adversarial], session_id=session_id)

    drain(det_worker, turns)
    drain(q_worker, turns)
    drain(r_worker, turns)
    drain(d_worker, turns)
    drain(inc_worker, turns)

    listener = FeedbackListener(redis_client, config)
    for tr in turn_results:
        listener.wait_for_outcome(tr.pipeline_message.incident_id, timeout_ms=2000, poll_block_ms=100)

    escalated_count = listener.session_escalation_count(session_id)
    print(f"\nRepeated-incident experiment: {escalated_count}/{turns} incidents escalated in session '{session_id}'")

    incident_manager = inc_worker.incident_manager
    threshold_triggered = False
    for tr in turn_results:
        snapshot = incident_manager.get_incident(tr.pipeline_message.incident_id)
        if snapshot and snapshot.escalated and any("REPEATED_INCIDENT" in r for r in snapshot.escalation_reasons):
            threshold_triggered = True
            break

    assert threshold_triggered, "expected REPEATED_INCIDENT_THRESHOLD_EXCEEDED to fire organically within this session"
