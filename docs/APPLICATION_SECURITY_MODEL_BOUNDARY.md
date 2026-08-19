# Application-Security Model Boundary Decision (Stage B)

## The question

Can the existing Phase 1-8 models (`RandomForest`/`XGBoost`/`IsolationForest`
ensemble, QSVM, VQC) legitimately consume `ApplicationFeatureVector`
(Stage B's new application-level feature representation)?

## The answer: No.

This is a factual, verifiable conclusion, not a stylistic preference.

`transform_sample()` (`src/preprocessing/classical_pipeline.py`) requires
exactly the 41 real NSL-KDD feature columns, verbatim, as dict keys --
`duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`,
`wrong_fragment`, `urgent`, `num_failed_logins`, `num_root`, `srv_count`,
`serror_rate`, and so on. These are TCP/UDP/ICMP network-flow statistics
computed over a captured connection window.

`ApplicationFeatureVector` (`src/agents/application_features.py`) has 11
fields: `request_rate`, `failed_auth_rate`, `validation_failure_rate`,
`endpoint_switch_rate`, `repeated_resource_access_rate`,
`invalid_resource_rate`, `auth_failure_burst`, `crud_anomaly_score`,
`response_error_rate`, `latency_anomaly_score`, `session_action_entropy`.
These are HTTP-request-and-response-sequence statistics computed over an
authenticated application session.

There is no honest field-by-field correspondence between these two sets.
Attempting to zero-pad or arbitrarily map application features into 41
NSL-KDD slots would not produce a meaningful representation for a model
that was trained on real network-flow statistics -- it would produce
noise the model has never seen a genuine signal for, and any resulting
"detection" would be an artifact of the mapping, not a real capability.

**Decision: the existing trained models (`artifacts/models/classical/*`,
`artifacts/models/quantum/*`) are not modified, not retrained, and not
fed `ApplicationFeatureVector` data, directly or via a forced mapping.**

## What this means architecturally

```
NSL-KDD path (Phases 1-8, completely unchanged):
  KDDTrain+.txt row / agent-generated exemplar
      -> transform_sample() -> 41-feature vector
      -> EnsembleClassicalDetector -> QuantumRouter -> HybridPipeline
      -> RiskEngine -> DefenseEngine -> IncidentManager

Application-security path (Stage B, new, parallel, not yet fully wired):
  Suzume interaction -> ApplicationObservation
      -> compute_application_features() -> ApplicationFeatureVector
      -> [FUTURE: a separately-trained Application Detection Model]
      -> [FUTURE: the SAME architectural pattern -- optional quantum
          verification, hybrid decision, risk, defense, incident --
          reusing the pattern, not the trained model objects]
```

## What Stage B actually delivers on this front

Stage B builds and tests the feature-extraction layer
(`compute_application_features`) end-to-end against real telemetry from
real (locally-controlled and, where reachable, live) Suzume interactions.
It does **not** build or train a new Application Detection Model in this
stage -- per the explicit instruction to document the boundary decision
*before* training anything new, and because a legitimate model requires
labeled training data, a training methodology, and an evaluation
methodology that don't exist yet and shouldn't be improvised under this
stage's time budget.

**What a future Application Detection Model stage would need to define
first** (not built here, listed so the gap is explicit rather than
silently skipped):
- **Feature schema**: `ApplicationFeatureVector`'s 11 fields, as defined above.
- **Labels**: what constitutes "normal" vs. "anomalous" application
  behavior needs a real labeling methodology -- e.g., NormalAgent-driven
  sessions labeled normal, AdversarialAgent-driven sessions (bounded,
  safe patterns only) labeled anomalous, collected across many sessions.
- **Training data source**: `CONTROLLED_LOCAL_SUZUME` sessions primarily
  (safe to generate at volume); `REAL_SUZUME_INTERACTION` sessions only
  within the strict configured bounds already required for any real
  interaction.
- **Training methodology**: a conventional classifier (the architecture
  doesn't require it to be a deep model) trained on
  `ApplicationFeatureVector` inputs -- explicitly NOT reusing or
  fine-tuning the existing NSL-KDD-trained model objects.
- **Evaluation methodology**: held-out sessions, not the same sessions
  used for threshold-setting -- avoiding exactly the "do not train
  against the test deployment traffic and then claim generalization"
  failure mode called out explicitly in the Stage B brief.
- **Quantum verification**: once a real Application Detection Model
  exists, the SAME `QuantumRouter`/`QSVMVerifier`/`VQCVerifier` classes
  could plausibly be reused as verification-stage components (they
  operate on a routing decision and a feature vector via `RoutingDecision`
  produced by whatever classical detector calls them) -- but the
  feature-dimensionality assumptions inside `QuantumPCA`/the feature maps
  were fit against the 41-dim NSL-KDD-derived space and would need their
  own re-fitting against the 11-dim application-feature space, not a
  direct reuse. This is real, identified follow-up work, not solved here.

This is the honest state of the model boundary at the end of Stage B:
the *representation* is real, tested, and grounded in actually-observed
telemetry; the *classifier* that would consume it for real detection
does not exist yet, and building it prematurely (without labels, without
an evaluation methodology) would violate the same evaluation-discipline
rule this project has followed since Phase 4.
