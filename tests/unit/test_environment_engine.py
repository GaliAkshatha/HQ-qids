import pytest

from src.agents.adversarial_agent import AdversarialAgent
from src.agents.environment_engine import EnvironmentEngine
from src.agents.normal_agent import NormalUserAgent
from src.agents.scenario_catalog import EnvironmentPolicy, ScenarioCatalog
from src.agents.session_correlation import SESSION_DELIMITER
from src.agents.templates import ExemplarBank


class StubGateway:
    def __init__(self):
        self.ingested = []

    def ingest(self, sample_id, raw_sample):
        self.ingested.append((sample_id, raw_sample))

        class FakeMessage:
            pass

        msg = FakeMessage()
        msg.correlation_id = sample_id.split(SESSION_DELIMITER)[0]
        msg.incident_id = f"inc-{len(self.ingested)}"
        return msg


@pytest.fixture(scope="module")
def catalog(repo_root):
    return ScenarioCatalog.load(repo_root / "config" / "agent_scenarios.json")


@pytest.fixture(scope="module")
def policy(repo_root):
    return EnvironmentPolicy.load(repo_root / "config" / "agent_environment_policy.json")


@pytest.fixture(scope="module")
def exemplar_bank(repo_root):
    return ExemplarBank(repo_root / "Data" / "raw" / "KDDTrain+.txt")


def test_run_produces_correct_number_of_turns(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=42)
    results = engine.run(turns=10)
    assert len(results) == 10
    assert len(gateway.ingested) == 10


def test_all_turns_in_one_run_share_session_id_by_default(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=1)
    results = engine.run(turns=6)
    session_ids = {r.session_id for r in results}
    assert len(session_ids) == 1


def test_sample_ids_are_unique_within_a_session(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=2)
    results = engine.run(turns=15)
    sample_ids = [r.sample_id for r in results]
    assert len(sample_ids) == len(set(sample_ids))


def test_generated_samples_carry_full_provenance(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=3)
    results = engine.run(turns=4)
    for r in results:
        assert r.record.sample_id == r.sample_id
        assert r.record.scenario_name in catalog.names()
        assert r.record.intended_label in ("normal", "attack")
        assert r.record.source_exemplar_index >= 0
        assert r.record.perturbation_magnitude == policy.perturbation_default


def test_rejects_perturbation_outside_configured_bounds(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=4)
    with pytest.raises(ValueError):
        engine.run(turns=1, perturbation=0.99)


def test_adversarial_agent_traffic_only_uses_allowed_scenarios(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    adversarial = AdversarialAgent("adv-test", policy.adversarial_allowed_scenarios)
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=5)
    results = engine.run(turns=20, agents=[adversarial])
    for r in results:
        assert r.record.scenario_name in policy.adversarial_allowed_scenarios
        assert r.record.intended_label == "attack"


def test_normal_agent_traffic_only_produces_normal_intent(catalog, policy, exemplar_bank):
    gateway = StubGateway()
    normal = NormalUserAgent("norm-test", policy.normal_allowed_scenarios)
    engine = EnvironmentEngine(gateway, catalog, policy, exemplar_bank=exemplar_bank, seed=6)
    results = engine.run(turns=10, agents=[normal])
    for r in results:
        assert r.record.intended_label == "normal"


def test_deterministic_with_fixed_seed(catalog, policy, exemplar_bank):
    """Confirms agents are deterministic/policy-based, not LLM-nondeterministic."""
    gateway1 = StubGateway()
    engine1 = EnvironmentEngine(gateway1, catalog, policy, exemplar_bank=exemplar_bank, seed=99)
    results1 = engine1.run(turns=5, session_id="fixed-session")

    gateway2 = StubGateway()
    engine2 = EnvironmentEngine(gateway2, catalog, policy, exemplar_bank=exemplar_bank, seed=99)
    results2 = engine2.run(turns=5, session_id="fixed-session")

    scenarios1 = [r.record.scenario_name for r in results1]
    scenarios2 = [r.record.scenario_name for r in results2]
    assert scenarios1 == scenarios2
