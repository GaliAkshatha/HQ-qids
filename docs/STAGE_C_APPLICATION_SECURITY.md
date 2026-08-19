# Stage C: Application-Security Detection + Adaptive Feedback

## Feature space

11 features from `src/agents/application_features.py` (Stage B):
`request_rate`, `failed_auth_rate`, `validation_failure_rate`,
`endpoint_switch_rate`, `repeated_resource_access_rate`,
`invalid_resource_rate`, `auth_failure_burst`, `crud_anomaly_score`,
`response_error_rate`, `latency_anomaly_score`, `session_action_entropy`.
Each is fully documented inline in that module. Small by design so
classical -> quantum PCA reduction stays computationally practical.

## Dataset-generation method

`src/agents/application_dataset.py`: real `SuzumeNormalAgent`/
`SuzumeAdversarialAgent` sessions run against the local controlled
Suzume-compatible target, each session's observations windowed into one
`ApplicationFeatureVector`. Labeled `AGENT_GENERATED_LABELED_DATA`.

## Labeling methodology and its limitation

Labels come from which agent generated the session, never from a
detector's own prediction. This is a legitimate bootstrapping approach,
but reflects known-generator identity, not independently-reviewed real
attack evidence. Adversarial scenarios are also deliberately crisp and
bounded rather than adversarially evasive -- this makes the dataset
close to trivially separable, which is exactly what the results below
show. Near-perfect classifier accuracy here reflects the boundedness of
these agent-generated scenarios, not evidence of real-world
attacker-evasion resistance.

## Train/test separation

`split_by_session()` splits by `session_id` (60/20/20), proven disjoint
by direct set-intersection assertion at split time and a dedicated
regression test. No window from one session appears in more than one
split. No threshold or hyperparameter was tuned against the test split.

## Real measurements (80 real sessions: 40 normal / 40 adversarial, 48/16/16 split)

### Classical baseline
| Model | Test Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8125 | 1.0 | 0.667 | 0.80 | 0.0 | 0.333 |
| Random Forest | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |

### Quantum comparison, bounded
2 qubits (PCA-reduced from 11 scaled application features), reusing the
existing quantum architecture with NEW artifacts -- never the
NSL-KDD-trained objects.

| Model | Test Accuracy | F1 | Train time | Inference (ms/sample) |
|---|---|---|---|---|
| QSVM | 1.0 | 1.0 | 3.4s | 144.0 |
| VQC | 0.9375 | 0.941 | 5.0s | 2.1 |

Quantum does not demonstrate improved detection here. QSVM matches the
classical Random Forest's perfect score; VQC is slightly lower. This is
consistent with the dataset's near-trivial separability, not evidence
about quantum verification's value in general. This is a bounded,
exploratory experiment on a small (80-session) dataset, not a
scientifically powered comparison -- no significance claim is made.

## Integration with existing architecture

`src/agents/application_pipeline.py` wires
`ApplicationSecurityDetector -> QuantumRouter -> HybridPipeline ->
DefenseEngine -> IncidentManager.record_full_lifecycle()` -- reusing
Phases 3-6 completely unmodified, with application-specific artifacts.
`record_full_lifecycle()` is the correct integration seam because it
accepts pre-computed contract objects without internally calling the
NSL-KDD-specific `transform_sample()`.

Two clearly separate sources exist and are never mixed: `HISTORICAL_NSLKDD`
and `APPLICATION_SECURITY`. `DetectionResult.metadata["source"]` tags
every application-security detection explicitly.

## Adaptive feedback

`src/agents/application_adaptation.py`: deterministic rule -- if the
previous session escalated, the adversarial agent switches away from
that scenario next session; otherwise it repeats it. Every decision is a
plain `AdaptationRecord`, fully inspectable, no LLM.

## What is genuinely novel vs. existing technology

Novel: the 11-feature application-behavior representation, the
session-level agent-generated labeled dataset with leakage-safe
splitting, and the demonstration that Phases 3-6's architecture (built
for NSL-KDD) genuinely generalizes to a second detection domain via
contract reuse alone.

Not novel, and not claimed to be: LogisticRegression/RandomForest are
standard sklearn classifiers; QSVM/VQC reuse the exact quantum machinery
built in Phase 2. No claim of improved real-world attack detection, no
claim of quantum advantage, no claim of generalization beyond this
80-session controlled dataset.

## Limitations

- Small dataset (80 sessions) -- not statistically powered.
- Labels reflect known-generator identity, not independent attack review.
- Adversarial scenarios are bounded/crisp, not evasive.
- No real Suzume interaction data used (network egress still blocked in
  this environment) -- entirely `CONTROLLED_LOCAL_SUZUME`.
- Quantum comparison is exploratory/bounded, not a significance test.
