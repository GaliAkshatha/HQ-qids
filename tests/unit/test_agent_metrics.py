from src.agents.metrics import AgentRunMetrics


def make_metrics_with_rows():
    m = AgentRunMetrics()
    for _ in range(3):
        m.record("normal", "normal_browsing", "normal", False, "MONITOR", 0.05, "normal", "sess1")
    for _ in range(3):
        m.record("attack", "neptune_flood", "attack", True, "ISOLATE_SIMULATED_SOURCE", 0.05, "adversarial", "sess1")
    m.record("attack", "neptune_flood", "normal", False, "MONITOR", 0.05, "adversarial", "sess1")
    return m


def test_confusion_matrix_counts_correctly():
    m = make_metrics_with_rows()
    matrix = m.confusion_matrix()
    assert matrix["normal"]["normal"] == 3
    assert matrix["normal"]["attack"] == 0
    assert matrix["attack"]["attack"] == 3
    assert matrix["attack"]["normal"] == 1


def test_confusion_matrix_accuracy():
    m = make_metrics_with_rows()
    acc = m.confusion_matrix_accuracy()
    assert acc == 6 / 7


def test_escalation_rate_by_scenario():
    m = make_metrics_with_rows()
    rates = m.escalation_rate_by_scenario()
    assert rates["normal_browsing"] == 0.0
    assert rates["neptune_flood"] == 3 / 4


def test_defense_action_distribution_by_scenario():
    m = make_metrics_with_rows()
    dist = m.defense_action_distribution_by_scenario()
    assert dist["normal_browsing"] == {"MONITOR": 3}
    assert dist["neptune_flood"] == {"ISOLATE_SIMULATED_SOURCE": 3, "MONITOR": 1}


def test_scenario_distribution():
    m = make_metrics_with_rows()
    dist = m.scenario_distribution()
    assert dist["normal_browsing"] == 3
    assert dist["neptune_flood"] == 4


def test_accuracy_by_perturbation_magnitude():
    m = AgentRunMetrics()
    m.record("attack", "neptune_flood", "attack", True, "ISOLATE_SIMULATED_SOURCE", 0.0, "adversarial", "s1")
    m.record("attack", "neptune_flood", "attack", True, "ISOLATE_SIMULATED_SOURCE", 0.0, "adversarial", "s1")
    m.record("attack", "neptune_flood", "normal", False, "MONITOR", 0.3, "adversarial", "s1")
    m.record("attack", "neptune_flood", "attack", True, "ISOLATE_SIMULATED_SOURCE", 0.3, "adversarial", "s1")

    result = m.accuracy_by_perturbation_magnitude()
    assert result[0.0] == 1.0
    assert result[0.3] == 0.5


def test_confusion_matrix_handles_missing_predictions_as_unknown():
    m = AgentRunMetrics()
    m.record("attack", "neptune_flood", None, None, None, 0.05, "adversarial", "s1")
    matrix = m.confusion_matrix()
    assert matrix["attack"]["unknown"] == 1


def test_total_and_observed_sample_counts():
    m = AgentRunMetrics()
    m.record("normal", "normal_browsing", "normal", False, "MONITOR", 0.05, "normal", "s1")
    m.record("attack", "neptune_flood", None, None, None, 0.05, "adversarial", "s1")
    assert m.total_samples() == 2
    assert m.samples_with_observed_outcome() == 1


def test_empty_metrics_do_not_crash():
    m = AgentRunMetrics()
    assert m.confusion_matrix_accuracy() is None
    assert m.escalation_rate_by_scenario() == {}
    assert m.defense_action_distribution_by_scenario() == {}
    assert m.total_samples() == 0
