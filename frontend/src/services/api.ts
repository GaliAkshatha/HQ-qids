// VITE_API_URL: absolute backend URL for production (e.g.
// "https://qids-api.onrender.com"). When unset, falls back to the
// existing relative "/api" path -- unchanged local-dev behavior, which
// works via vite.config.ts's dev-server proxy to localhost:8080.
const API_BASE = `${(import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? ""}/api`;

export interface Experiment {
  experiment_id: string;
  scenario: string;
  mode: string;
  n_sessions: number;
  quantum: string;
  status: string;
  created_at: string;
  completed_at: string | null;
  incident_count: number;
  error: string | null;
  incidents?: IncidentSummary[];
}

export interface IncidentSummary {
  incident_id: string;
  correlation_id: string;
  current_state: string;
  escalated: boolean;
  scenario: string;
  agent_type: string;
  experiment_id: string;
  session_index: number;
}

export interface IncidentSnapshot {
  incident_id: string;
  correlation_id: string;
  current_state: string;
  created_at: string;
  updated_at: string;
  escalated: boolean;
  escalation_reasons: string[];
}

export interface TimelineEvent {
  event_type: string;
  previous_state: string | null;
  new_state: string | null;
  reason: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface IncidentDetail {
  incident: IncidentSnapshot;
  timeline: TimelineEvent[];
}

export interface SystemStatus {
  api: boolean;
  redis: boolean;
  classical_detector: boolean;
  vqc: boolean;
  qsvm: boolean;
  application_detector: boolean;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  false_negative_rate: number;
  training_time_seconds?: number;
  inference_time_ms_per_sample?: number;
}

export interface MetricsResponse {
  system_status: SystemStatus;
  pipeline_metrics: {
    incident: Record<string, unknown>;
    defense: Record<string, unknown>;
    hybrid: Record<string, unknown>;
  };
  model_comparison: {
    dataset_label: string;
    bounded_experiment?: boolean;
    train_size?: number;
    test_size?: number;
    models: Record<string, ModelMetrics>;
  };
}

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => req<{ status: string }>("/health"),
  ready: () => req<{ status: string; checks: SystemStatus }>("/ready"),

  startExperiment: (params: { scenario: string; n_sessions: number; mode: string; quantum: string }) =>
    req<Experiment>("/experiments/start", { method: "POST", body: JSON.stringify(params) }),
  listExperiments: () => req<{ experiments: Experiment[]; scenarios: string[] }>("/experiments"),
  getExperiment: (id: string) => req<Experiment>(`/experiments/${id}`),

  listAgents: () => req<{ agents: { agent_type: string; sessions: number; escalated: number; scenarios: string[] }[] }>("/agents"),

  listIncidents: () => req<{ incidents: IncidentSnapshot[] }>("/incidents"),
  getIncident: (id: string) => req<IncidentDetail>(`/incidents/${id}`),

  getMetrics: () => req<MetricsResponse>("/metrics"),

  eventSource: () => new EventSource(`${API_BASE}/events`),
};
