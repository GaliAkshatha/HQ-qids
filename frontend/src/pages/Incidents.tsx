import { useEffect, useState } from "react";
import { api, IncidentSnapshot, IncidentDetail } from "../services/api";
import { IncidentStateBadge } from "../components/Badges";
import { EmptyState, ErrorBanner, LoadingSkeleton } from "../components/StateBlocks";

export default function Incidents() {
  const [incidents, setIncidents] = useState<IncidentSnapshot[] | null>(null);
  const [detail, setDetail] = useState<IncidentDetail | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    api.listIncidents().then((r) => setIncidents(r.incidents)).catch((e) => setError(e.message));
  }, []);

  const openIncident = (id: string) => {
    setSelectedId(id);
    setDetailError(null);
    api.getIncident(id).then(setDetail).catch((e) => setDetailError(e.message));
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <h1>Incidents</h1>
          <p>Every incident tracked by the QIDS incident lifecycle, with its complete evidence timeline.</p>
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      <div className="card">
        {incidents === null ? (
          <LoadingSkeleton lines={4} />
        ) : incidents.length === 0 ? (
          <EmptyState icon="●" title="No incidents yet" detail="Run an experiment from the Experiments page to generate real incidents." />
        ) : (
          <table>
            <thead>
              <tr><th>Incident ID</th><th>State</th><th>Escalated</th><th>Updated</th></tr>
            </thead>
            <tbody>
              {incidents.map((inc) => (
                <tr key={inc.incident_id} data-clickable="true" onClick={() => openIncident(inc.incident_id)}>
                  <td><code>{inc.incident_id.slice(0, 8)}</code></td>
                  <td><IncidentStateBadge state={inc.current_state} /></td>
                  <td>{inc.escalated ? "Yes" : "No"}</td>
                  <td>{new Date(inc.updated_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selectedId && (
        <>
          <h2>Incident Timeline</h2>
          <div className="card">
            {detailError && <ErrorBanner message={detailError} />}
            {!detail && !detailError && <LoadingSkeleton lines={3} />}
            {detail && (
              <>
                <div style={{ marginBottom: 16 }}>
                  <h3>{detail.incident.incident_id}</h3>
                  <p>Correlation ID: <code>{detail.incident.correlation_id}</code></p>
                  <div className="status-row" style={{ marginTop: 8 }}>
                    <IncidentStateBadge state={detail.incident.current_state} />
                    {detail.incident.escalated && (
                      <span>— {detail.incident.escalation_reasons.join(", ")}</span>
                    )}
                  </div>
                </div>
                {detail.timeline.map((event, i) => (
                  <div className="timeline-item" key={i}>
                    <div>
                      <span className="timeline-event-type">{event.event_type}</span>
                      {event.previous_state && event.new_state && (
                        <span className="timeline-transition">{event.previous_state} → {event.new_state}</span>
                      )}
                    </div>
                    <div className="timeline-reason">{event.reason}</div>
                    <div className="timeline-time">{new Date(event.timestamp).toLocaleString()}</div>
                  </div>
                ))}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
