import { useEffect, useState } from "react";
import { api, Experiment } from "../services/api";
import { EmptyState, ErrorBanner, LoadingSkeleton } from "../components/StateBlocks";

function ExperimentStatusBadge({ status }: { status: string }) {
  const cls = status === "completed" ? "badge-resolved" : status === "failed" ? "badge-escalated" : "badge-neutral";
  return <span className={`badge ${cls}`}>{status}</span>;
}

export default function Experiments() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [scenario, setScenario] = useState("normal_browsing");
  const [nSessions, setNSessions] = useState(5);
  const [mode, setMode] = useState("normal");
  const [quantum, setQuantum] = useState("auto");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const refresh = () => api.listExperiments().then((r) => {
    setScenarios(r.scenarios);
    setExperiments(r.experiments);
    setLoaded(true);
  }).catch((e) => { setError(e.message); setLoaded(true); });

  useEffect(() => { refresh(); }, []);

  const start = async () => {
    setRunning(true);
    setError(null);
    try {
      await api.startExperiment({ scenario, n_sessions: nSessions, mode, quantum });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  if (!loaded) return <div><div className="page-header"><h1>Live Agent Lab</h1></div><LoadingSkeleton lines={3} /></div>;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Live Agent Lab</h1>
          <p>Send bounded, controlled telemetry through the QIDS pipeline and observe real detection outcomes.</p>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Configure Experiment</div>
        <div className="grid-4">
          <div>
            <label>Scenario</label>
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {scenarios.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label>Sessions (max 50)</label>
            <input type="number" min={1} max={50} value={nSessions} onChange={(e) => setNSessions(Number(e.target.value))} />
          </div>
          <div>
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="normal">Normal</option>
              <option value="adversarial">Adversarial</option>
              <option value="mixed">Mixed</option>
            </select>
          </div>
          <div>
            <label>Quantum</label>
            <select value={quantum} onChange={(e) => setQuantum(e.target.value)}>
              <option value="auto">Auto</option>
              <option value="disabled">Disabled</option>
            </select>
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <button onClick={start} disabled={running}>{running ? "Running…" : "Start Session"}</button>
        </div>
        {error && <ErrorBanner message={error} />}
        <div className="note-banner">
          Traffic is bounded, agent-generated telemetry (perturbed real NSL-KDD exemplars) sent through QIDS as an authorized application would — not arbitrary internet traffic and not real-world attack generation.
        </div>
      </div>

      <h2>Experiment History</h2>
      <div className="card">
        {experiments.length === 0 ? (
          <EmptyState icon="▷" title="No experiments run yet" detail="Configure an experiment above and click Start Session." />
        ) : (
          <table>
            <thead>
              <tr><th>ID</th><th>Scenario</th><th>Mode</th><th>Sessions</th><th>Status</th><th>Incidents</th></tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={exp.experiment_id}>
                  <td><code>{exp.experiment_id.slice(0, 8)}</code></td>
                  <td>{exp.scenario}</td>
                  <td>{exp.mode}</td>
                  <td>{exp.n_sessions}</td>
                  <td><ExperimentStatusBadge status={exp.status} /></td>
                  <td>{exp.incident_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
