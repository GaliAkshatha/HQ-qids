"""
The core Phase 7 proof: real NSL-KDD rows through traffic-gateway ->
Redis Streams -> detection-worker -> quantum-worker (real VQC) ->
risk-worker -> defense-worker -> incident-worker -> persisted event
history, using the actual service build_worker() factories (the same
code that would run in the Docker containers), each driven by explicit
run_once() calls for deterministic, single-threaded test control.
"""

import dataclasses
import json
import uuid

import pytest
import redis

from src.preprocessing.classical_pipeline import load_raw
from src.runtime.config import RuntimePolicyConfig
from src.runtime.services import defense_worker, detection_worker, incident_worker, quantum_worker, risk_worker, traffic_gateway
from src.runtime.tracing import trace_correlation


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
    streams = {k: f"{v}.test-{suffix}" for k, v in base.streams.items()}
    groups = {k: f"{v}.test-{suffix}" for k, v in base.consumer_groups.items()}
    return dataclasses.replace(base, streams=streams, consumer_groups=groups)


@pytest.fixture(scope="module")
def real_raw_rows(repo_root):
    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    return df.drop(columns=["label", "difficulty"]).iloc[:15]


def run_full_pipeline_for_n_samples(redis_client, config, raw_rows, event_store_path):
    gateway = traffic_gateway.TrafficGateway(redis_client, config)

    det_worker = detection_worker.build_worker(config)
    q_worker = quantum_worker.build_worker(config)
    r_worker = risk_worker.build_worker(config)
    d_worker = defense_worker.build_worker(config)
    inc_worker = incident_worker.build_worker(config, event_store_path=event_store_path)

    ingested = []
    for i in range(len(raw_rows)):
        sample = raw_rows.iloc[i].to_dict()
        msg = gateway.ingest(f"dist-e2e-{i}", sample)
        ingested.append(msg)

    for _ in range(len(raw_rows)):
        det_worker.run_once(block_ms=2000)
    for _ in range(len(raw_rows)):
        q_worker.run_once(block_ms=2000)
    for _ in range(len(raw_rows)):
        r_worker.run_once(block_ms=2000)
    for _ in range(len(raw_rows)):
        d_worker.run_once(block_ms=2000)
    for _ in range(len(raw_rows)):
        inc_worker.run_once(block_ms=2000)

    return {
        "gateway": gateway, "ingested": ingested,
        "detection_worker": det_worker, "quantum_worker": q_worker, "risk_worker": r_worker,
        "defense_worker": d_worker, "incident_worker": inc_worker,
    }


def test_real_distributed_pipeline_end_to_end(redis_client, real_raw_rows, tmp_path):
    config = make_isolated_config()
    event_store_path = tmp_path / "distributed_events.jsonl"

    result = run_full_pipeline_for_n_samples(redis_client, config, real_raw_rows, event_store_path)

    for stream_key in [
        config.streams["traffic_ingested"], config.streams["detection_completed"],
        config.streams["quantum_completed"], config.streams["risk_assessed"],
        config.streams["defense_completed"], config.streams["incident_updated"],
    ]:
        entries = redis_client.xrange(stream_key, min="-", max="+")
        assert len(entries) == len(real_raw_rows), f"{stream_key} has {len(entries)} entries, expected {len(real_raw_rows)}"

    incident_manager = result["incident_worker"].incident_manager
    for msg in result["ingested"]:
        snapshot = incident_manager.get_incident_by_correlation(msg.correlation_id)
        assert snapshot is not None
        assert snapshot.is_terminal
        assert snapshot.current_state in ("RESOLVED", "ESCALATED")

    print("\nReal distributed incident metrics:", incident_manager.metrics_snapshot())


def test_correlation_causation_incident_id_lineage_is_preserved(redis_client, real_raw_rows, tmp_path):
    config = make_isolated_config()
    event_store_path = tmp_path / "distributed_events_lineage.jsonl"

    result = run_full_pipeline_for_n_samples(redis_client, config, real_raw_rows.iloc[:3], event_store_path)

    for msg in result["ingested"]:
        correlation_id = msg.correlation_id
        incident_id = msg.incident_id

        all_messages = []
        for stream_key in [
            config.streams["traffic_ingested"], config.streams["detection_completed"],
            config.streams["quantum_completed"], config.streams["risk_assessed"],
            config.streams["defense_completed"], config.streams["incident_updated"],
        ]:
            entries = redis_client.xrange(stream_key, min="-", max="+")
            for _id, fields in entries:
                data = json.loads(fields["data"])
                if data["correlation_id"] == correlation_id:
                    all_messages.append(data)

        assert len(all_messages) == 6
        for m in all_messages:
            assert m["correlation_id"] == correlation_id
            assert m["incident_id"] == incident_id

        event_ids = {m["event_id"] for m in all_messages}
        root_count = 0
        for m in all_messages:
            if m["causation_id"] is None:
                root_count += 1
            else:
                assert m["causation_id"] in event_ids, "causation_id must point to a real prior event in this lineage"
        assert root_count == 1


def test_trace_utility_reconstructs_real_lineage(redis_client, real_raw_rows, tmp_path):
    config = make_isolated_config()
    event_store_path = tmp_path / "distributed_events_trace.jsonl"
    result = run_full_pipeline_for_n_samples(redis_client, config, real_raw_rows.iloc[:1], event_store_path)

    correlation_id = result["ingested"][0].correlation_id
    entries = trace_correlation(correlation_id)
    assert len(entries) > 0
    services_seen = {e.service for e in entries}
    assert "traffic_gateway" in services_seen or any("gateway" in s for s in services_seen)
