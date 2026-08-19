import { useEffect, useState } from "react";
import { api } from "../services/api";
import { AgentTypeBadge } from "../components/Badges";
import { EmptyState, ErrorBanner, LoadingSkeleton } from "../components/StateBlocks";

interface AgentSummary {
  agent_type: string;
  sessions: number;
  escalated: number;
  scenarios: string[];
}

export default function Agents() {
  const [agents, setAgents] = useState<AgentSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listAgents().then((r) => setAgents(r.agents)).catch((e) => setError(e.message));
  }, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Agents</h1>
          <p>Normal and adversarial agents generate bounded, controlled application behavior for testing detection.</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="note-banner" style={{ marginBottom: 24 }}>
        Agents generate bounded, controlled application telemetry (perturbed real NSL-KDD exemplars) sent through QIDS as an authorized application would — not arbitrary internet traffic.
      </div>

      {agents === null ? (
        <LoadingSkeleton lines={2} />
      ) : agents.length === 0 ? (
        <div className="card">
          <EmptyState icon="◆" title="No agent activity yet" detail="Run an experiment from the Experiments page to see agent sessions here." />
        </div>
      ) : (
        <div className="grid">
          {agents.map((agent) => (
            <div className="card" key={agent.agent_type}>
              <div style={{ marginBottom: 12 }}><AgentTypeBadge agentType={agent.agent_type} /></div>
              <div className="metric-tile" style={{ marginBottom: 12 }}>
                <span className="metric-value">{agent.sessions}</span>
                <span className="metric-label">Sessions Run</span>
              </div>
              <p><strong>{agent.escalated}</strong> escalated</p>
              <p style={{ fontSize: 12.5 }}>Scenarios: {agent.scenarios.join(", ")}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
