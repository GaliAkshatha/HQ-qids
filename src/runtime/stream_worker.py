"""
src/runtime/stream_worker.py

The one place consume-loop/ACK/retry/backoff/DLQ/idempotency/pending-
recovery logic lives. Every worker subclasses this and implements only
handle() (call the domain component) and output_stream() (where to
publish the result) -- no worker reimplements any of this plumbing.

Two distinct reliability mechanisms, kept separate on purpose:
  1. Handled failure (handle() raises): retry-with-backoff by
     republishing an incremented-retry_count copy to the SAME input
     stream, ACKing the original (it's being intentionally superseded).
     Exceeding max_retries -> dead letter.
  2. Worker crash (process dies mid-processing, before ACK): the message
     stays genuinely pending in Redis. recover_pending() uses XAUTOCLAIM
     to reclaim entries idle past min_idle_time_ms and reprocess them --
     this is what proves "a worker crash must not silently lose an
     event," tested by killing a worker before it ACKs.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

import redis

from src.contracts import PipelineMessage
from src.runtime.config import RuntimePolicyConfig
from src.runtime.dead_letter import publish_to_dead_letter
from src.observability.logging_config import log_event


class StreamWorker(ABC):
    def __init__(
        self,
        name: str,
        client: redis.Redis,
        config: RuntimePolicyConfig,
        input_stream: str,
        consumer_group: str,
        idempotency_store,
        logger,
    ) -> None:
        self.name = name
        self.client = client
        self.config = config
        self.input_stream = input_stream
        self.group = consumer_group
        self.consumer_name = f"{config.consumer_name_prefix}-{name}-{uuid.uuid4().hex[:8]}"
        self.idempotency = idempotency_store
        self.logger = logger
        self._running = False
        self._ensure_group()

    # ---- required by subclasses: pure domain translation, no plumbing ----

    @abstractmethod
    def handle(self, message: PipelineMessage) -> Optional[PipelineMessage]:
        """Process one message by calling the relevant domain component
        and return the next PipelineMessage to publish (or None if this
        worker is terminal). Must raise on failure -- never swallow."""
        raise NotImplementedError

    @abstractmethod
    def output_stream(self) -> Optional[str]:
        raise NotImplementedError

    # ---- consumer group setup -------------------------------------------------

    def _ensure_group(self) -> None:
        try:
            self.client.xgroup_create(self.input_stream, self.group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # ---- publish ----------------------------------------------------------------

    def _publish(self, stream_key: str, message: PipelineMessage) -> str:
        return self.client.xadd(stream_key, {"data": json.dumps(message.to_dict(), default=str)})

    # ---- core processing ----------------------------------------------------------

    def run_once(self, block_ms: Optional[int] = None) -> bool:
        """Reads and processes one batch. Returns True if any entry was seen."""
        block_ms = self.config.block_ms if block_ms is None else block_ms
        response = self.client.xreadgroup(
            self.group, self.consumer_name, {self.input_stream: ">"},
            count=self.config.batch_size, block=block_ms,
        )
        if not response:
            return False
        processed_any = False
        for _stream_key, entries in response:
            for entry_id, fields in entries:
                self._process_entry(entry_id, fields)
                processed_any = True
        return processed_any

    def _process_entry(self, entry_id: str, fields: dict) -> None:
        message = PipelineMessage.from_dict(json.loads(fields["data"]))

        if self.idempotency.already_processed(message.event_id):
            log_event(
                self.logger, 20, "duplicate delivery detected -- skipped, no reprocessing",
                correlation_id=message.correlation_id, causation_id=message.causation_id,
                incident_id=message.incident_id, event_type=message.event_type, event_id=message.event_id,
            )
            self.client.xack(self.input_stream, self.group, entry_id)
            return

        t0 = time.perf_counter()
        try:
            result = self.handle(message)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if result is not None and self.output_stream():
                self._publish(self.output_stream(), result)

            # mark_processed happens AFTER the side effect + publish
            # succeed, immediately before ACK -- see idempotency.py's
            # documented ordering contract.
            self.idempotency.mark_processed(message.event_id)
            self.client.xack(self.input_stream, self.group, entry_id)

            log_event(
                self.logger, 20, "event processed", correlation_id=message.correlation_id,
                causation_id=message.causation_id, incident_id=message.incident_id,
                event_type=message.event_type, event_id=message.event_id, latency_ms=latency_ms,
            )
        except Exception as e:  # noqa: BLE001 -- deliberate: route through retry/DLQ, never crash the loop
            latency_ms = (time.perf_counter() - t0) * 1000.0
            log_event(
                self.logger, 40, f"event processing failed: {e}", correlation_id=message.correlation_id,
                causation_id=message.causation_id, incident_id=message.incident_id,
                event_type=message.event_type, event_id=message.event_id, latency_ms=latency_ms,
                retry_count=message.retry_count,
            )
            self._handle_failure(entry_id, message, str(e))

    def _handle_failure(self, entry_id: str, message: PipelineMessage, error: str) -> None:
        if message.retry_count < self.config.max_retries:
            backoff = self.config.backoff_seconds * (self.config.backoff_multiplier ** message.retry_count)
            time.sleep(backoff)
            retried = message.next_retry()
            self._publish(self.input_stream, retried)
            self.client.xack(self.input_stream, self.group, entry_id)
            log_event(
                self.logger, 30, f"retrying (attempt {retried.retry_count}/{self.config.max_retries})",
                correlation_id=message.correlation_id, event_id=message.event_id, event_type=message.event_type,
            )
        else:
            publish_to_dead_letter(self.client, self.config, message, error, self.input_stream, self.name)
            self.client.xack(self.input_stream, self.group, entry_id)
            log_event(
                self.logger, 40, "retries exhausted -- sent to dead letter",
                correlation_id=message.correlation_id, event_id=message.event_id, event_type=message.event_type,
            )

    # ---- pending-message recovery (worker-crash safety) --------------------------

    def recover_pending(self) -> List[str]:
        """Reclaims entries left pending by a crashed consumer (never
        ACKed) that have been idle past min_idle_time_ms, and reprocesses
        them through the exact same _process_entry() path."""
        claimed_ids: List[str] = []
        cursor = "0-0"
        while True:
            result = self.client.xautoclaim(
                self.input_stream, self.group, self.consumer_name,
                min_idle_time=self.config.min_idle_time_ms, start_id=cursor, count=self.config.claim_batch_size,
            )
            cursor, entries = result[0], result[1]
            if not entries:
                break
            for entry_id, fields in entries:
                log_event(self.logger, 30, "recovered pending entry via XAUTOCLAIM", entry_id=entry_id)
                self._process_entry(entry_id, fields)
                claimed_ids.append(entry_id)
            if cursor == "0-0":
                break
        return claimed_ids

    # ---- run loop -----------------------------------------------------------------

    def run_forever(self, max_iterations: Optional[int] = None) -> None:
        self._running = True
        iterations = 0
        while self._running:
            self.recover_pending()
            self.run_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

    def stop(self) -> None:
        self._running = False
