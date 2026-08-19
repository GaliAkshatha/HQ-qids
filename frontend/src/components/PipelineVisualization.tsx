import { StreamEvent } from "../services/useEventStream";

const STAGES: { key: string; label: string; icon: string; eventTypes: string[] }[] = [
  { key: "agent", label: "Agent", icon: "◆", eventTypes: [] },
  { key: "detection", label: "Detection", icon: "◈", eventTypes: ["DETECTION_COMPLETED"] },
  { key: "quantum", label: "Quantum", icon: "◇", eventTypes: ["QUANTUM_ROUTING", "QUANTUM_VERIFICATION"] },
  { key: "risk", label: "Risk", icon: "▲", eventTypes: ["HYBRID_DECISION", "RISK_ASSESSED"] },
  { key: "defense", label: "Defense", icon: "■", eventTypes: ["DEFENSE_EXECUTED"] },
  { key: "incident", label: "Incident", icon: "●", eventTypes: ["INCIDENT_UPDATED"] },
];

interface Props {
  events: StreamEvent[];
}

/**
 * Every count and every "active" highlight below is derived directly
 * from real SSE events (which themselves come from real, persisted
 * IncidentEvent records -- see src/api/services/event_service.py). No
 * value here is decorative or fabricated; with zero events, every stage
 * shows a real zero, not a placeholder animation.
 */
export default function PipelineVisualization({ events }: Props) {
  const mostRecentType = events[0]?.event_type;

  const counts: Record<string, number> = {};
  for (const stage of STAGES) {
    counts[stage.key] = stage.eventTypes.reduce(
      (sum, t) => sum + events.filter((e) => e.event_type === t).length,
      0
    );
  }
  counts["agent"] = events.length > 0 ? events.length : 0;

  return (
    <div className="pipeline-flow">
      {STAGES.map((stage, i) => {
        const isActive = stage.key !== "agent" && stage.eventTypes.includes(mostRecentType ?? "");
        return (
          <div key={stage.key} style={{ display: "flex", alignItems: "center" }}>
            <div className={`pipeline-stage${isActive ? " active" : ""}`}>
              <span className="stage-icon">{stage.icon}</span>
              <span className="stage-name">{stage.label}</span>
              <span className="stage-count">{counts[stage.key]} events</span>
            </div>
            {i < STAGES.length - 1 && <span className="pipeline-arrow">→</span>}
          </div>
        );
      })}
    </div>
  );
}
