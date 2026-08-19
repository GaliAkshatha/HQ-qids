export function IncidentStateBadge({ state }: { state: string }) {
  const cls = state === "RESOLVED" ? "badge-resolved" : state === "ESCALATED" ? "badge-escalated" : "badge-neutral";
  return <span className={`badge ${cls}`}>{state}</span>;
}

export function AgentTypeBadge({ agentType }: { agentType: string }) {
  const cls = agentType === "adversarial" ? "badge-attack" : "badge-normal";
  return <span className={`badge ${cls}`}>{agentType}</span>;
}

export function RiskBadge({ level }: { level: string }) {
  const cls = `badge-risk-${level.toLowerCase()}`;
  return <span className={`badge ${cls}`}>{level}</span>;
}

export function StatusDot({ up }: { up: boolean }) {
  return <span className={`status-dot ${up ? "status-up" : "status-down"}`} />;
}
