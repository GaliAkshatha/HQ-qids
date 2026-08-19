"""
src/api/services/experiment_service.py

Orchestrates bounded experiments by reusing EXISTING, unmodified
components: NormalUserAgent/AdversarialAgent + templates.generate_sample
(Phase 8), EnsembleClassicalDetector, QuantumRouter, HybridPipeline,
DefenseEngine, IncidentManager (Phases 1-6). Runs synchronously via
IncidentManager.process() -- simpler and reliable for a request/response
API; the distributed Redis-worker path (Phase 7) remains available and
is started by run.py for architectural completeness, but is not required
for the API's own experiment execution.

No detection/quantum/risk/defense/incident logic lives here -- only
orchestration and result shaping.
"""

from __future__ import annotations

import os
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.agents.adversarial_agent import AdversarialAgent
from src.agents.normal_agent import NormalUserAgent
from src.agents.scenario_catalog import EnvironmentPolicy, ScenarioCatalog
from src.agents.templates import ExemplarBank, generate_sample
from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import EventStore, JsonlEventStore
from src.incident.incident_manager import IncidentManager
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter
from src.runtime.config import RuntimePolicyConfig
from src.runtime.redis_client import build_redis_client

MAX_SESSIONS = 50
VALID_MODES = {"normal", "adversarial", "mixed"}
VALID_QUANTUM = {"auto", "disabled"}

DEFAULT_EVENT_STORE_PATH = "logs/api_incident_events.jsonl"


def _build_event_store() -> EventStore:
    """
    EVENT_STORE_BACKEND (default "jsonl") selects between the two
    EXISTING EventStore implementations -- JsonlEventStore (Phase 6) and
    RedisEventStore (Phase 7), neither modified here. EVENT_STORE_PATH
    (default "logs/api_incident_events.jsonl") overrides the JSONL file
    location, useful on platforms where only a specific directory is
    writable/persistent.
    """
    backend = os.environ.get("EVENT_STORE_BACKEND", "jsonl").lower()
    if backend == "redis":
        from src.runtime.event_store_redis import RedisEventStore

        config = RuntimePolicyConfig.load()
        client = build_redis_client(config)
        return RedisEventStore(client, key_prefix="api_incident_event_store")
    path = os.environ.get("EVENT_STORE_PATH", DEFAULT_EVENT_STORE_PATH)
    return JsonlEventStore(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Experiment:
    experiment_id: str
    scenario: str
    mode: str
    n_sessions: int
    quantum: str
    status: str = "pending"
    created_at: str = field(default_factory=_now_iso)
    completed_at: Optional[str] = None
    incidents: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id, "scenario": self.scenario, "mode": self.mode,
            "n_sessions": self.n_sessions, "quantum": self.quantum, "status": self.status,
            "created_at": self.created_at, "completed_at": self.completed_at,
            "incident_count": len(self.incidents), "error": self.error,
        }


class ExperimentService:
    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()
        self._catalog = ScenarioCatalog.load()
        self._policy = EnvironmentPolicy.load()
        self._exemplar_bank = ExemplarBank()
        self._build_engine_components()

    def _build_engine_components(self) -> None:
        self._detector = EnsembleClassicalDetector.load(
            models_dir="artifacts/models/classical", preprocessing_dir="artifacts/preprocessing",
        )
        routing_policy = RoutingPolicyConfig.load()
        verifier = VQCVerifier.load(models_dir="artifacts/models/quantum/vqc", preprocessing_dir="artifacts/preprocessing")
        self._router = QuantumRouter(policy=routing_policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=4))
        self._hybrid_pipeline = HybridPipeline()
        self._defense_engine = DefenseEngine(policy=DefensePolicyConfig.load())
        self._event_store = _build_event_store()
        self._incident_manager = IncidentManager(
            detector=self._detector, router=self._router, hybrid_pipeline=self._hybrid_pipeline,
            defense_engine=self._defense_engine, event_store=self._event_store,
            escalation_policy=EscalationPolicyConfig.load(),
        )

    def list_scenarios(self) -> List[str]:
        return self._catalog.names()

    def create_experiment(self, scenario: str, n_sessions: int, mode: str, quantum: str) -> Experiment:
        if scenario not in self._catalog.names():
            raise ValueError(f"Unknown scenario: '{scenario}'. Known: {self._catalog.names()}")
        if not (1 <= n_sessions <= MAX_SESSIONS):
            raise ValueError(f"n_sessions must be between 1 and {MAX_SESSIONS}")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        if quantum not in VALID_QUANTUM:
            raise ValueError(f"quantum must be one of {VALID_QUANTUM}")

        experiment = Experiment(experiment_id=str(uuid.uuid4()), scenario=scenario, mode=mode, n_sessions=n_sessions, quantum=quantum)
        with self._lock:
            self._experiments[experiment.experiment_id] = experiment
        return experiment

    def run_experiment(self, experiment_id: str) -> Experiment:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Unknown experiment_id: '{experiment_id}'")

        experiment.status = "running"
        rng = random.Random(hash(experiment_id) % (2**32))
        try:
            agents = self._build_agents(experiment.scenario, experiment.mode)
            for i in range(experiment.n_sessions):
                agent = agents[i % len(agents)]
                action = agent.act(f"{experiment.experiment_id}-sess-{i}", rng)
                scenario_def = self._catalog.get(action.scenario_name)

                raw_sample, source_idx, source_label, perturbed = generate_sample(
                    scenario_def, self._exemplar_bank, self._policy.perturbation_default, rng,
                )
                sample_id = f"{experiment.experiment_id}::{i}::{uuid.uuid4().hex[:8]}"
                snapshot = self._incident_manager.process(sample_id, raw_sample)
                experiment.incidents.append({
                    "incident_id": snapshot.incident_id, "correlation_id": snapshot.correlation_id,
                    "current_state": snapshot.current_state, "escalated": snapshot.escalated,
                    "scenario": action.scenario_name, "agent_type": action.agent_type,
                    "experiment_id": experiment.experiment_id, "session_index": i,
                })

            experiment.status = "completed"
            experiment.completed_at = _now_iso()
        except Exception as e:  # noqa: BLE001
            experiment.status = "failed"
            experiment.error = str(e)
        return experiment

    def _build_agents(self, scenario: str, mode: str) -> List:
        scenario_def = self._catalog.get(scenario)
        if mode == "normal" or scenario_def.category == "normal":
            return [NormalUserAgent("exp-normal", (scenario,))]
        if mode == "adversarial":
            return [AdversarialAgent("exp-adversarial", (scenario,))]
        normal_scenarios = tuple(s for s in self._catalog.names() if self._catalog.get(s).category == "normal")
        return [NormalUserAgent("exp-normal", normal_scenarios), AdversarialAgent("exp-adversarial", (scenario,))]

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        with self._lock:
            return list(self._experiments.values())

    def get_incident(self, incident_id: str):
        return self._incident_manager.get_incident(incident_id)

    def get_events(self, incident_id: str):
        return self._incident_manager.get_events(incident_id)

    def list_incidents(self) -> List:
        all_events = self._event_store.read_all()
        incident_ids = list({e.incident_id for e in all_events})
        return [self._incident_manager.get_incident(i) for i in incident_ids if self._incident_manager.get_incident(i)]

    def metrics_snapshot(self) -> Dict:
        return {
            "incident": self._incident_manager.metrics_snapshot(),
            "defense": self._defense_engine.metrics.snapshot(),
            "hybrid": self._hybrid_pipeline.metrics_snapshot(),
        }
