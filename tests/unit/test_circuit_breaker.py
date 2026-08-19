from src.routing.circuit_breaker import CLOSED, HALF_OPEN, OPEN, QuantumCircuitBreaker


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_breaker(failure_threshold=3, cooldown_seconds=30, clock=None):
    clock = clock or FakeClock()
    return QuantumCircuitBreaker(failure_threshold, cooldown_seconds, now_fn=clock), clock


def test_starts_closed_and_allows_requests():
    breaker, _ = make_breaker()
    assert breaker.state("VQC") == CLOSED
    assert breaker.allow_request("VQC") is True


def test_successes_keep_it_closed():
    breaker, _ = make_breaker()
    for _ in range(10):
        breaker.record_success("VQC")
    assert breaker.state("VQC") == CLOSED


def test_failures_below_threshold_stay_closed():
    breaker, _ = make_breaker(failure_threshold=3)
    breaker.record_failure("VQC")
    breaker.record_failure("VQC")
    assert breaker.state("VQC") == CLOSED


def test_failures_at_threshold_open_the_circuit():
    breaker, _ = make_breaker(failure_threshold=3)
    breaker.record_failure("VQC")
    breaker.record_failure("VQC")
    breaker.record_failure("VQC")
    assert breaker.state("VQC") == OPEN
    assert breaker.allow_request("VQC") is False
    assert breaker.is_available("VQC") is False


def test_success_resets_consecutive_failure_count():
    breaker, _ = make_breaker(failure_threshold=3)
    breaker.record_failure("VQC")
    breaker.record_failure("VQC")
    breaker.record_success("VQC")
    breaker.record_failure("VQC")
    breaker.record_failure("VQC")
    assert breaker.state("VQC") == CLOSED  # count reset by the success, only at 2 again


def test_open_before_cooldown_denies_requests():
    breaker, clock = make_breaker(failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("VQC")
    assert breaker.state("VQC") == OPEN
    clock.advance(29.9)
    assert breaker.allow_request("VQC") is False
    assert breaker.state("VQC") == OPEN


def test_open_after_cooldown_transitions_to_half_open_and_allows_one_trial():
    breaker, clock = make_breaker(failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("VQC")
    clock.advance(30.1)
    assert breaker.allow_request("VQC") is True
    assert breaker.state("VQC") == HALF_OPEN


def test_half_open_denies_concurrent_second_trial():
    breaker, clock = make_breaker(failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("VQC")
    clock.advance(30.1)
    assert breaker.allow_request("VQC") is True  # claims the trial
    assert breaker.allow_request("VQC") is False  # second caller denied, trial already in flight
    assert breaker.state("VQC") == HALF_OPEN


def test_half_open_success_closes_circuit_and_resets():
    breaker, clock = make_breaker(failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("VQC")
    clock.advance(30.1)
    breaker.allow_request("VQC")
    breaker.record_success("VQC")
    assert breaker.state("VQC") == CLOSED
    snap = breaker.snapshot("VQC")
    assert snap["consecutive_failures"] == 0
    assert snap["half_open_trial_in_flight"] is False


def test_half_open_failure_reopens_and_restarts_cooldown():
    breaker, clock = make_breaker(failure_threshold=1, cooldown_seconds=30)
    breaker.record_failure("VQC")
    clock.advance(30.1)
    breaker.allow_request("VQC")
    breaker.record_failure("VQC")
    assert breaker.state("VQC") == OPEN
    # cooldown restarted -- immediately after re-opening, still not available
    assert breaker.allow_request("VQC") is False
    clock.advance(30.1)
    assert breaker.allow_request("VQC") is True  # cooldown elapsed again


def test_backends_are_tracked_independently():
    breaker, _ = make_breaker(failure_threshold=1)
    breaker.record_failure("QSVM")
    assert breaker.state("QSVM") == OPEN
    assert breaker.state("VQC") == CLOSED
    assert breaker.allow_request("VQC") is True
