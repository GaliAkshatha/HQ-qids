"""
src/routing/job_queue.py

In-process asynchronous job execution, backed by a ThreadPoolExecutor
(approved Decision 1). Deliberately generic -- this module knows nothing
about quantum verification, circuit breakers, or routing policy; it just
runs submitted callables asynchronously and tracks timing. The router
builds a quantum-aware job on top of this.

Kept behind this narrow interface (submit / get_result / get_status /
get_queue_wait_time_ms) specifically so Phase 7 can later provide a
Redis-Streams-or-RabbitMQ-backed implementation of the same shape without
the router changing.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, Optional


PENDING = "PENDING"
RUNNING = "RUNNING"
DONE = "DONE"


class QuantumJobQueue:
    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[str, Future] = {}
        self._submit_times: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        submit_time = time.monotonic()

        def _wrapped():
            with self._lock:
                self._start_times[job_id] = time.monotonic()
            return fn(*args, **kwargs)

        future = self._executor.submit(_wrapped)
        with self._lock:
            self._futures[job_id] = future
            self._submit_times[job_id] = submit_time
        return job_id

    def get_result(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """
        Blocks up to `timeout` seconds. Returns whatever the submitted
        callable returned once done. Returns None (not an exception) if
        the job is still running when the wait times out -- callers
        distinguish "still pending" from "failed" by inspecting the
        returned value's own status, not by catching an exception here.
        """
        future = self._require_future(job_id)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            return None

    def get_status(self, job_id: str) -> str:
        future = self._require_future(job_id)
        if future.done():
            return DONE
        with self._lock:
            started = job_id in self._start_times
        return RUNNING if started else PENDING

    def get_queue_wait_time_ms(self, job_id: str) -> Optional[float]:
        with self._lock:
            submit_t = self._submit_times.get(job_id)
            start_t = self._start_times.get(job_id)
        if submit_t is None or start_t is None:
            return None
        return (start_t - submit_t) * 1000.0

    def _require_future(self, job_id: str) -> Future:
        future = self._futures.get(job_id)
        if future is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return future

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
