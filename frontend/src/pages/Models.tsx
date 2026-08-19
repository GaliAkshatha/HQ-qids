import { useEffect, useState } from "react";
import { api, MetricsResponse } from "../services/api";
import { LoadingSkeleton, ErrorBanner, EmptyState } from "../components/StateBlocks";

export default function Models() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getMetrics().then(setMetrics).catch((e) => setError(e.message));
  }, []);

  if (error) return <div><div className="page-header"><h1>Model Comparison</h1></div><ErrorBanner message={error} /></div>;
  if (!metrics) return <div><div className="page-header"><h1>Model Comparison</h1></div><LoadingSkeleton lines={3} /></div>;

  const { model_comparison } = metrics;
  const modelNames = Object.keys(model_comparison.models);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Model Comparison</h1>
          <p>Classical and quantum-assisted application-security detectors, evaluated on the same held-out split.</p>
        </div>
      </div>

      <div className="note-banner" style={{ marginBottom: 24 }}>
        Dataset: <strong>{model_comparison.dataset_label}</strong>. Train/test sizes: {model_comparison.train_size ?? "n/a"} / {model_comparison.test_size ?? "n/a"} sessions.
        {model_comparison.bounded_experiment && " This is a bounded, exploratory experiment, not a statistically powered comparison."}
        {" "}These are not production-validated attack-detection accuracy figures.
      </div>

      <div className="card">
        {modelNames.length === 0 ? (
          <EmptyState icon="▦" title="No model comparison data available" detail="Model results are generated during offline training runs (see docs/STAGE_C_APPLICATION_SECURITY.md)." />
        ) : (
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th><th>FNR</th>
                <th>Train time (s)</th><th>Inference (ms/sample)</th>
              </tr>
            </thead>
            <tbody>
              {modelNames.map((name) => {
                const m = model_comparison.models[name];
                return (
                  <tr key={name}>
                    <td style={{ fontWeight: 600 }}>{name}</td>
                    <td>{m.accuracy.toFixed(3)}</td>
                    <td>{m.precision.toFixed(3)}</td>
                    <td>{m.recall.toFixed(3)}</td>
                    <td>{m.f1.toFixed(3)}</td>
                    <td>{m.false_positive_rate.toFixed(3)}</td>
                    <td>{m.false_negative_rate.toFixed(3)}</td>
                    <td>{m.training_time_seconds?.toFixed(2) ?? "—"}</td>
                    <td>{m.inference_time_ms_per_sample?.toFixed(1) ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
