# Incident Identity Model: `incident_id` vs `correlation_id`

**Status:** Resolved during Phase 8 implementation. This document is the
authoritative reference for this distinction going forward.

## Background

Through Phase 7, `IncidentManager` was built and tested exclusively under
`SampleIdCorrelation`, where `correlation_id` is always derived directly
from `sample_id`. Because every NSL-KDD row has a unique `sample_id`, this
meant `correlation_id` and `incident_id` were *always* in a strict 1:1
relationship in practice -- a fact that was true by construction, but was
never made an explicit, separately-enforced rule anywhere in the code.
`IncidentManager`'s terminal-incident idempotency check reflected this
implicit assumption: it stored at most one `IncidentSnapshot` per
correlation key.

Phase 8 introduced `AgentSessionCorrelation`, which deliberately breaks
that 1:1 assumption on purpose -- multiple distinct agent turns
(different, real, independently-generated samples) are meant to share
one session/correlation key, specifically so Phase 6's
`repeated_incident_threshold` escalation rule could be exercised
organically for the first time. Running the real distributed
repeated-incident experiment (multiple adversarial turns, one session)
exposed the implicit assumption directly: after the first incident under
a shared key reached a terminal state, `IncidentManager` treated every
subsequent, genuinely distinct incident under that same key as a
duplicate delivery of the first one, and silently skipped processing it
entirely. Four of five real generated samples in the first repeated-
incident test run were never processed at all.

## The resolved model

**`incident_id` is the idempotency identity.**
A redelivered `incident_id` -- whether from a Redis consumer-group
redelivery, a retried `process()` call, or a process restart -- must be
detected and skipped. It must never re-run detection, quantum
verification, defense, or incident lifecycle processing a second time.

**`correlation_id` is the grouping/correlation identity.**
Multiple distinct `incident_id`s are explicitly allowed to share one
`correlation_id`. Each is processed fully and independently. The shared
`correlation_id` is used for session/entity-level history
(`get_incidents_by_correlation()`) and for repeated-incident counting
(`RepeatedIncidentTracker`, keyed by correlation key).

**Under `SampleIdCorrelation` (the Phase 1-7 default, unchanged), the two
remain effectively 1:1** -- the general model above simply specializes to
the old behavior when `correlation_id == sample_id`; nothing about
existing single-incident-per-key semantics changed.

## What changed in `IncidentManager` (`src/incident/incident_manager.py`)

- `self._snapshots` is now keyed by `incident_id`, not `correlation_id`.
- A new `self._correlation_index: Dict[correlation_key, List[incident_id]]`
  tracks the one-to-many relationship explicitly.
- A new `self._sample_id_index: Dict[sample_id, incident_id]` gives
  `process()` its own natural idempotency key (it always mints a fresh
  `incident_id` itself, so "redelivery of the same `incident_id`" isn't a
  concept that applies to that entry point the way it does to the
  distributed `incident-worker`, which receives `incident_id` from an
  upstream message and can genuinely see it redelivered).
- `get_incident_by_correlation()` is preserved, unchanged in return type
  and behavior for every existing caller (returns the most recent
  incident for a key) -- under `SampleIdCorrelation` this is identical to
  the old behavior, since there's only ever one.
- A new `get_incidents_by_correlation()` returns the full, ordered list --
  the correct view once multiple distinct incidents can share one key.
- `src/runtime/services/incident_worker.py`'s idempotency check was
  updated to call `get_incident(incident_id)` directly, rather than
  `get_incident_by_correlation(correlation_key)` -- this is the actual
  fix; the `IncidentManager` internals above are what make it possible.
- `record_full_lifecycle()` itself performs no idempotency check -- both
  callers (`process()` via `_run_full_chain()`, and
  `incident_worker.handle()`) gate before calling it. This means
  `RepeatedIncidentTracker.record_incident()` (called inside
  `record_full_lifecycle()`) is only ever reached for genuinely new
  incidents, never for duplicates -- a duplicate delivery cannot inflate
  the repeated-incident count.

## Why this was an approved exception to "do not modify Phase 1-7 domain logic"

Every other Phase 8 constraint held: no ML/quantum/defense logic was
touched, no contract was modified, and the change is additive to
`IncidentManager`'s internals (existing method signatures and return
types are unchanged for every Phase 1-7 caller). The change was made
because Phase 8's approved `AgentSessionCorrelation` extension exposed a
genuine incompatibility between an implicit Phase 6 assumption and an
explicitly approved Phase 8 capability -- not because Phase 8 needed
convenience or because the architecture was being generally redesigned.

## Regression proof

`tests/unit/test_incident_correlation_identity_distinction.py` contains
six tests (A-E plus a backward-compatibility test) proving this model
directly: duplicate `incident_id` is skipped exactly once; distinct
`incident_id`s sharing one `correlation_id` both process independently;
three distinct incidents cross the repeated-incident threshold
organically; a duplicate delivery among them does not inflate the count;
the same holds after a full `JsonlEventStore` restart/reconstruction; and
`SampleIdCorrelation` is proven to remain strictly 1:1.

`tests/integration/test_agent_distributed_pipeline.py`'s
`test_repeated_adversarial_turns_organically_trigger_repeated_incident_escalation`
proves the same thing against the real distributed pipeline: five real,
independently-generated adversarial samples, one real session
correlation key, real Redis, real workers -- `REPEATED_INCIDENT_THRESHOLD_EXCEEDED`
fires organically on the 3rd, 4th, and 5th incidents.
