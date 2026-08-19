"""
The mandatory persistence/restart test:
  1. Creates an incident.
  2. Processes it to a terminal state.
  3. Creates a new IncidentManager instance.
  4. Loads the same JsonlEventStore.
  5. Reconstructs the incident.
  6. Calls process() again.
  7. Confirms no downstream detection/quantum/defense processing occurs.
  8. Records the appropriate IDEMPOTENT_SKIP event.

Uses a real JsonlEventStore file on disk (not InMemoryEventStore) and
real VQC artifacts, so this is a genuine restart proof, not a simulated
one.
"""

from src.defense.defense_engine import DefenseEngine
from src.defense.defense_policy import DefensePolicyConfig
from src.detection.ensemble_detector import EnsembleClassicalDetector
from src.hybrid.pipeline import HybridPipeline
from src.incident.escalation import EscalationPolicyConfig
from src.incident.event_store import JsonlEventStore
from src.incident.incident_manager import IncidentManager
from src.preprocessing.classical_pipeline import load_raw
from src.quantum.vqc_verifier import VQCVerifier
from src.routing.job_queue import QuantumJobQueue
from src.routing.policy import RoutingPolicyConfig
from src.routing.router import QuantumRouter

CLASSICAL_MODELS = "artifacts/models/classical"
VQC_MODELS = "artifacts/models/quantum/vqc"
PREPROCESSING = "artifacts/preprocessing"


class CountingDetector:
    """Wraps the real detector but counts calls, so we can prove step 7
    (no downstream detection occurs on the idempotent replay) without
    mocking away real detection logic for the first call."""

    def __init__(self, real_detector):
        self._real = real_detector
        self.call_count = 0
        self.preprocessing = real_detector.preprocessing

    def detect(self, sample, sample_id=None):
        self.call_count += 1
        return self._real.detect(sample, sample_id=sample_id)


def build_manager(repo_root, event_store, counting_detector):
    routing_policy = RoutingPolicyConfig.load(repo_root / "config" / "routing_policy.json")
    verifier = VQCVerifier.load(models_dir=repo_root / VQC_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    router = QuantumRouter(policy=routing_policy, verifier=verifier, job_queue=QuantumJobQueue(max_workers=2))
    hybrid_pipeline = HybridPipeline()
    defense_engine = DefenseEngine(policy=DefensePolicyConfig.load(repo_root / "config" / "defense_policy.json"))
    escalation_policy = EscalationPolicyConfig.load(repo_root / "config" / "incident_policy.json")

    return IncidentManager(
        detector=counting_detector, router=router, hybrid_pipeline=hybrid_pipeline, defense_engine=defense_engine,
        event_store=event_store, escalation_policy=escalation_policy,
    )


def test_restart_reconstructs_terminal_incident_and_prevents_reprocessing(repo_root, tmp_path):
    real_detector = EnsembleClassicalDetector.load(models_dir=repo_root / CLASSICAL_MODELS, preprocessing_dir=repo_root / PREPROCESSING)
    event_store_path = tmp_path / "incident_events.jsonl"

    df = load_raw(repo_root / "Data" / "raw" / "KDDTrain+.txt")
    sample = df.drop(columns=["label", "difficulty"]).iloc[0].to_dict()

    # ---- 1 & 2: create and process an incident to a terminal state ----
    detector_1 = CountingDetector(real_detector)
    store_1 = JsonlEventStore(event_store_path)
    manager_1 = build_manager(repo_root, store_1, detector_1)

    original_snapshot = manager_1.process("restart-test-sample", sample)
    assert original_snapshot.is_terminal
    assert detector_1.call_count == 1
    original_event_count = len(manager_1.get_events(original_snapshot.incident_id))

    # tear down manager_1 entirely -- nothing about it persists except the file
    del manager_1
    del detector_1
    del store_1

    # ---- 3 & 4: brand-new IncidentManager instance, loading the SAME
    #      JsonlEventStore file from disk ----
    detector_2 = CountingDetector(real_detector)
    store_2 = JsonlEventStore(event_store_path)  # fresh instance, reads the file on construction
    manager_2 = build_manager(repo_root, store_2, detector_2)

    # ---- 5: reconstructs the incident ----
    reconstructed = manager_2.get_incident_by_correlation("restart-test-sample")
    assert reconstructed is not None
    assert reconstructed.incident_id == original_snapshot.incident_id
    assert reconstructed.current_state == original_snapshot.current_state
    assert reconstructed.is_terminal
    assert reconstructed.event_ids == original_snapshot.event_ids

    # ---- 6: calls process() again ----
    replay_snapshot = manager_2.process("restart-test-sample", sample)

    # ---- 7: confirms no downstream detection/quantum/defense processing occurs ----
    assert detector_2.call_count == 0  # the new process's detector was NEVER invoked
    assert replay_snapshot.incident_id == original_snapshot.incident_id
    assert replay_snapshot.current_state == original_snapshot.current_state

    # ---- 8: records the appropriate IDEMPOTENT_SKIP event ----
    events_after_replay = manager_2.get_events(original_snapshot.incident_id)
    assert len(events_after_replay) == original_event_count + 1
    assert events_after_replay[-1].event_type == "IDEMPOTENT_SKIP"

    # and the file on disk reflects it too -- a THIRD fresh instance sees it
    store_3 = JsonlEventStore(event_store_path)
    events_from_disk = store_3.read_all(correlation_id="restart-test-sample")
    assert events_from_disk[-1].event_type == "IDEMPOTENT_SKIP"
    assert len(events_from_disk) == original_event_count + 1
