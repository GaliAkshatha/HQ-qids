"""
The required final demonstration: kill one worker before it ACKs a
message, restart it, use XAUTOCLAIM to recover the pending event, and
prove the pipeline continues to a correct terminal state WITHOUT losing
the event or duplicating the defense side effect.

"Kill" here means: read the message via xreadgroup (creating a genuine
pending entry in Redis, exactly as a real consumer would), then
deliberately do NOT process or ACK it -- simulating the process dying at
that exact point. A second, independently-constructed worker instance
(standing in for "the container restarted") then calls recover_pending(),
which uses XAUTOCLAIM to reclaim it and process it for real.
"""

import dataclasses
import time
import uuid

import pytest
import redis

from src.preprocessing.classical_pipeline import load_raw
from src.runtime.config import RuntimePolicyConfig
from src.runtime.services import defense_worker as defense_worker_mod
from src.runtime.services import detection_worker, incident_worker, quantum_worker as quantum_worker_mod, risk_worker, traffic_gateway


@pytest.fixture()
def redis_client():
    client = redis.Redis(decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("real redis-server not available in this environment")
    yield client


def make_isolated_config(min_idle_time_ms=100) -> RuntimePolicyConfig:
    base = RuntimePolicyConfig.load()
    suffix = uuid.uuid4().hex[:8]
    streams = {k: f"{v}.crash-test-{suffix}" for k, v in base.streams.items()}
    groups = {k: f"{v}.crash-test-{suffix}" for k, v in base.consumer_groups.items()}
    return dataclasses.replace(base, streams=streams, consumer_groups=groups, min_idle_time_ms=min_idle_time_ms)


def test_defense_worker_crash_before_ack_recovers_via_xautoclaim_no_duplicate_side_effect(redis_client, repo_root, tmp_path):
    config = make_isolated_config(min_idle_time_ms=100)

    gateway = traffic_gateway.TrafficGateway(redis_client, config)
    det_worker = detection_worker.build_worker(config)
    q_worker = quantum_worker_mod.build_worker(config)
    r_worker = risk_worker.build_worker(config)

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[2].to_dict()  # real HIGH_ANOMALY row
    msg = gateway.ingest("crash-recovery-sample", sample)

    det_worker.run_once(block_ms=3000)
    q_worker.run_once(block_ms=5000)
    r_worker.run_once(block_ms=3000)

    # ---- simulate defense-worker crashing BEFORE it ACKs ----
    crashed_worker = defense_worker_mod.build_worker(config)
    input_stream = config.streams["risk_assessed"]
    group = config.consumer_groups["defense_worker"]

    response = redis_client.xreadgroup(group, crashed_worker.consumer_name, {input_stream: ">"}, count=10)
    assert response, "expected the risk.assessed message to be available"

    pending = redis_client.xpending(input_stream, group)
    assert pending["pending"] == 1  # genuinely stuck -- the "crash" is real

    assert crashed_worker.defense_engine.metrics.snapshot()["actions_executed"] == 0

    time.sleep(0.15)  # exceed min_idle_time_ms

    # ---- "restart": a brand new defense-worker instance ----
    restarted_worker = defense_worker_mod.build_worker(config)
    claimed = restarted_worker.recover_pending()
    assert len(claimed) == 1  # the pending entry was recovered via XAUTOCLAIM

    assert restarted_worker.defense_engine.metrics.snapshot()["actions_executed"] == 1
    assert crashed_worker.defense_engine.metrics.snapshot()["actions_executed"] == 0  # crashed instance never ran it

    pending_after = redis_client.xpending(input_stream, group)
    assert pending_after["pending"] == 0

    # ---- pipeline continues: incident-worker completes the chain ----
    inc_worker = incident_worker.build_worker(config, event_store_path=tmp_path / "crash_recovery_events.jsonl")
    inc_worker.run_once(block_ms=3000)

    snapshot = inc_worker.incident_manager.get_incident_by_correlation("crash-recovery-sample")
    assert snapshot is not None
    assert snapshot.is_terminal
    print(f"\nCrash-recovery demonstration: incident reached {snapshot.current_state} after XAUTOCLAIM recovery, defense action executed exactly once.")
