"""
Minimal real QSVM path through the FULL distributed pipeline (all 6
services, real Redis). Deliberately a single sample given the real
~2.1-2.4s/sample measured cost.
"""

import dataclasses
import uuid

import pytest
import redis

from src.preprocessing.classical_pipeline import load_preprocessing_artifacts, load_raw
from src.quantum.qsvm_verifier import QSVMVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter
from src.runtime.config import RuntimePolicyConfig
from src.runtime.services import defense_worker, detection_worker, incident_worker, risk_worker, traffic_gateway
from src.runtime.services.quantum_worker import QuantumWorker


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
    streams = {k: f"{v}.qsvm-test-{suffix}" for k, v in base.streams.items()}
    groups = {k: f"{v}.qsvm-test-{suffix}" for k, v in base.consumer_groups.items()}
    return dataclasses.replace(base, streams=streams, consumer_groups=groups)


def test_single_real_qsvm_sample_through_full_distributed_pipeline(redis_client, repo_root, tmp_path):
    config = make_isolated_config()

    qsvm_models_dir = repo_root / "artifacts" / "models" / "quantum" / "qsvm"
    preprocessing_dir = repo_root / "artifacts" / "preprocessing"

    gateway = traffic_gateway.TrafficGateway(redis_client, config)
    det_worker = detection_worker.build_worker(config)

    routing_policy = RoutingPolicyConfig.load().with_overrides(quantum_backend="QSVM")
    qsvm_verifier = QSVMVerifier.load(models_dir=qsvm_models_dir, preprocessing_dir=preprocessing_dir)
    qsvm_router = QuantumRouter(policy=routing_policy, verifier=qsvm_verifier, job_queue=QuantumJobQueue(max_workers=1))
    preprocessing_artifacts = load_preprocessing_artifacts(preprocessing_dir)
    q_worker = QuantumWorker(redis_client, config, qsvm_router, preprocessing_artifacts)

    r_worker = risk_worker.build_worker(config)
    d_worker = defense_worker.build_worker(config)
    inc_worker = incident_worker.build_worker(config, event_store_path=tmp_path / "qsvm_dist_events.jsonl")

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    # index 2 is a real row genuinely flagged HIGH_ANOMALY by the real
    # routing policy (verified separately, not forced via a synthetic
    # confidence override) -- so this test actually exercises QSVM,
    # rather than a row the policy correctly decides to skip.
    sample = df.drop(columns=["label", "difficulty"]).iloc[2].to_dict()

    msg = gateway.ingest("qsvm-dist-1", sample)

    det_worker.run_once(block_ms=3000)
    q_worker.run_once(block_ms=15000)  # real QSVM call, ~2.1-2.4s
    r_worker.run_once(block_ms=3000)
    d_worker.run_once(block_ms=3000)
    inc_worker.run_once(block_ms=3000)

    snapshot = inc_worker.incident_manager.get_incident_by_correlation(msg.correlation_id)
    assert snapshot is not None
    assert snapshot.is_terminal
    print("\nReal QSVM distributed incident:", snapshot.current_state, snapshot.event_ids)
