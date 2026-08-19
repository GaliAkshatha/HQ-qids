"""
src/agents/application_dataset.py

Generates AGENT_GENERATED_LABELED_DATA: real ApplicationObservation
telemetry from real SuzumeNormalAgent/SuzumeAdversarialAgent sessions
against the local controlled target, windowed into ApplicationFeatureVector
samples, labeled from the KNOWN agent/scenario ground truth (never from a
detector's own prediction).

LEAKAGE PREVENTION: splitting is done by session_id -- every window from
one session goes entirely into train, val, or test, never split across.

LIMITATION (documented, not hidden): labels come from which agent/
scenario generated the traffic, not from independent human review or a
real attack ground truth. This is a legitimate bootstrapping approach
for a first dataset, but not the same evidentiary strength as a labeled
real-world incident dataset -- results should not be read as
"real-world attack detection accuracy."
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

from src.agents.adversarial_agent import SUZUME_ADVERSARIAL_SCENARIOS, SuzumeAdversarialAgent
from src.agents.agent_session import AgentSession
from src.agents.application_features import ApplicationFeatureVector, compute_application_features
from src.agents.normal_agent import SuzumeNormalAgent
from src.agents.suzume_traffic_source import SuzumeTrafficSource

DATASET_LABEL = "AGENT_GENERATED_LABELED_DATA"


@dataclass
class LabeledSample:
    scenario_id: str
    session_id: str
    agent_type: str
    feature_window: ApplicationFeatureVector
    label: str
    dataset_label: str = DATASET_LABEL


def generate_labeled_sessions(
    traffic_source: SuzumeTrafficSource,
    n_normal_sessions: int,
    n_adversarial_sessions: int,
    seed: int = 0,
) -> List[LabeledSample]:
    rng = random.Random(seed)
    samples: List[LabeledSample] = []
    adversarial_scenario_ids = list(SUZUME_ADVERSARIAL_SCENARIOS.keys())

    for i in range(n_normal_sessions):
        agent = SuzumeNormalAgent(f"dataset-normal-{i}", seed=rng.randint(0, 1_000_000))
        session = AgentSession(
            session_id=str(uuid.uuid4()), agent_id=agent.agent_id, agent_type="normal",
            target_label="CONTROLLED_LOCAL_SUZUME", created_at=datetime.now(timezone.utc).isoformat(),
        )
        observations = agent.run_session(traffic_source, session)
        if not observations:
            continue
        vec = compute_application_features(observations)
        samples.append(LabeledSample(
            scenario_id="normal_workflow", session_id=session.session_id, agent_type="normal",
            feature_window=vec, label="normal",
        ))

    for i in range(n_adversarial_sessions):
        scenario_id = adversarial_scenario_ids[i % len(adversarial_scenario_ids)]
        agent = SuzumeAdversarialAgent(f"dataset-adv-{i}", allowed_scenario_ids=(scenario_id,), seed=rng.randint(0, 1_000_000))
        session = AgentSession(
            session_id=str(uuid.uuid4()), agent_id=agent.agent_id, agent_type="adversarial",
            target_label="CONTROLLED_LOCAL_SUZUME", created_at=datetime.now(timezone.utc).isoformat(),
        )
        observations = agent.run_session(traffic_source, session, scenario_id=scenario_id)
        if not observations:
            continue
        vec = compute_application_features(observations)
        samples.append(LabeledSample(
            scenario_id=scenario_id, session_id=session.session_id, agent_type="adversarial",
            feature_window=vec, label="anomalous",
        ))

    return samples


def split_by_session(
    samples: List[LabeledSample], train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0
) -> Tuple[List[LabeledSample], List[LabeledSample], List[LabeledSample]]:
    session_ids = list({s.session_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(session_ids)

    n = len(session_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = set(session_ids[:n_train])
    val_ids = set(session_ids[n_train:n_train + n_val])
    test_ids = set(session_ids[n_train + n_val:])

    train = [s for s in samples if s.session_id in train_ids]
    val = [s for s in samples if s.session_id in val_ids]
    test = [s for s in samples if s.session_id in test_ids]

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    return train, val, test


FEATURE_NAMES = [
    "request_rate", "failed_auth_rate", "validation_failure_rate", "endpoint_switch_rate",
    "repeated_resource_access_rate", "invalid_resource_rate", "auth_failure_burst",
    "crud_anomaly_score", "response_error_rate", "latency_anomaly_score", "session_action_entropy",
]


def to_matrix(samples: List[LabeledSample]):
    X = [[getattr(s.feature_window, name) for name in FEATURE_NAMES] for s in samples]
    y = [0 if s.label == "normal" else 1 for s in samples]
    return X, y
