import time

import pytest

from src.routing.job_queue import DONE, QuantumJobQueue


def test_submit_returns_a_job_id_and_result_is_retrievable():
    queue = QuantumJobQueue(max_workers=2)
    job_id = queue.submit(lambda x: x * 2, 21)
    result = queue.get_result(job_id, timeout=5)
    assert result == 42


def test_get_result_returns_none_if_still_pending_within_timeout():
    queue = QuantumJobQueue(max_workers=1)

    def slow():
        time.sleep(0.5)
        return "done"

    job_id = queue.submit(slow)
    result = queue.get_result(job_id, timeout=0.05)
    assert result is None  # still running, caller's wait budget expired

    # job keeps running in the background regardless -- eventually resolves
    final = queue.get_result(job_id, timeout=5)
    assert final == "done"


def test_get_status_transitions_to_done():
    queue = QuantumJobQueue(max_workers=1)
    job_id = queue.submit(lambda: 1)
    queue.get_result(job_id, timeout=5)
    assert queue.get_status(job_id) == DONE


def test_get_queue_wait_time_is_measured():
    queue = QuantumJobQueue(max_workers=1)
    job_id = queue.submit(lambda: 1)
    queue.get_result(job_id, timeout=5)
    wait_ms = queue.get_queue_wait_time_ms(job_id)
    assert wait_ms is not None
    assert wait_ms >= 0


def test_unknown_job_id_raises_key_error():
    queue = QuantumJobQueue(max_workers=1)
    with pytest.raises(KeyError):
        queue.get_result("does-not-exist", timeout=1)


def test_multiple_concurrent_jobs_all_resolve_correctly():
    queue = QuantumJobQueue(max_workers=4)
    job_ids = [queue.submit(lambda x=i: x * x) for i in range(10)]
    results = [queue.get_result(jid, timeout=5) for jid in job_ids]
    assert results == [i * i for i in range(10)]


def test_job_that_raises_propagates_exception_through_get_result():
    queue = QuantumJobQueue(max_workers=1)

    def boom():
        raise ValueError("intentional")

    job_id = queue.submit(boom)
    with pytest.raises(ValueError):
        queue.get_result(job_id, timeout=5)
