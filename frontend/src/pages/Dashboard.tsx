import { useEffect, useState } from "react";
import { api, MetricsResponse } from "../services/api";
import { useEventStream } from "../services/useEventStream";
import EventStream from "../components/EventStream";
import PipelineVisualization from "../components/PipelineVisualization";
import { StatusDot } from "../components/Badges";
import { LoadingSkeleton, ErrorBanner } from "../components/StateBlocks";

const STATUS_LABELS: Record<string, string> = {
  api: "API",
  redis: "Redis",
  classical_detector: "Classical Detector",
  vqc: "VQC (Quantum)",
  qsvm: "QSVM (Quantum)",
  application_detector: "Application Detector",
};

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { events, connectionState } = useEventStream();

  useEffect(() => {
    const load = () => api.getMetrics().then((m) => { setMetrics(m); setError(null); }).catch((e) => setError(e.message));
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  if (error && !metrics) {
    return (
      <div>
        <div className="page-header"><h1>Dashboard</h1></div>
        <ErrorBanner message={`Could not reach the QIDS API: ${error}`} />
      </div>
    );
  }
  if (!metrics) {
    return (
      <div>
        <div className="page-header"><h1>Dashboard</h1></div>
        <LoadingSkeleton lines={3} />
      </div>
    );
  }

  const { system_status, pipeline_metrics } = metrics;
  const incident = pipeline_metrics.incident as Record<string, number | null>;
  const hybrid = pipeline_metrics.hybrid as Record<string, number | null>;
  const defense = pipeline_metrics.defense as Record<string, number | null>;
  const hasActivity = (incident.total_incidents ?? 0) > 0;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Dashboard</h1>
          <p>Real-time status of the detection → quantum → risk → defense → incident pipeline.</p>
        </div>
      </div>

      {error && <ErrorBanner message={`Last metrics refresh failed: ${error}`} />}

      <h2>System Status</h2>
      <div className="card grid-metrics">
        {Object.entries(system_status).map(([key, up]) => (
          <div className="status-row" key={key}>
            <StatusDot up={up as boolean} />
            {STATUS_LABELS[key] ?? key}
          </div>
        ))}
      </div>

      <h2>Live Security Pipeline</h2>
      <div className="card">
        <PipelineVisualization events={events} />
      </div>

      <h2>Pipeline Metrics</h2>
      <div className="card grid-metrics">
        <div className="metric-tile">
          <span className="metric-value">{incident.total_incidents ?? 0}</span>
          <span className="metric-label">Total Incidents</span>
        </div>
        <div className="metric-tile">
          <span className="metric-value">{incident.resolved_incidents ?? 0}</span>
          <span className="metric-label">Resolved</span>
        </div>
        <div className="metric-tile">
          <span className="metric-value">{incident.escalated_incidents ?? 0}</span>
          <span className="metric-label">Escalated</span>
        </div>
        <div className="metric-tile">
          <span className="metric-value">{hybrid.quantum_invocations ?? 0}</span>
          <span className="metric-label">Quantum Invocations</span>
        </div>
        <div className="metric-tile">
          <span className="metric-value">
            {hybrid.quantum_confirmation_rate == null ? "—" : `${((hybrid.quantum_confirmation_rate as number) * 100).toFixed(0)}%`}
          </span>
          <span className="metric-label">Quantum Confirmation Rate</span>
        </div>
        <div className="metric-tile">
          <span className="metric-value">
            {defense.defense_success_rate == null ? "—" : `${((defense.defense_success_rate as number) * 100).toFixed(0)}%`}
          </span>
          <span className="metric-label">Defense Success Rate</span>
        </div>
      </div>
      {!hasActivity && (
        <div className="info-banner">
          No experiments have been run yet in this session. Head to the <strong>Experiments</strong> page to send a bounded, controlled experiment through the pipeline and see real activity here.
        </div>
      )}

      <h2>Live Event Stream</h2>
      <EventStream events={events} connectionState={connectionState} />
    </div>
  );
}
