import { StreamEvent, ConnectionState } from "../services/useEventStream";
import { EmptyState } from "./StateBlocks";

interface Props {
  events: StreamEvent[];
  connectionState: ConnectionState;
}

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting to live event stream…",
  connected: "Live — connected to event stream",
  disconnected: "Disconnected — attempting to reconnect",
};

export default function EventStream({ events, connectionState }: Props) {
  return (
    <div className="card">
      <div className="connection-indicator">
        <span className={`status-dot ${connectionState === "connected" ? "status-up" : connectionState === "connecting" ? "connecting" : "status-down"}`} />
        {CONNECTION_LABEL[connectionState]}
      </div>
      <div className="event-log">
        {events.length === 0 ? (
          <EmptyState icon="◇" title="No events yet" detail="Run an experiment to see real pipeline events appear here." />
        ) : (
          events.map((e, i) => (
            <div className="event-row" key={i}>
              <span className="event-time">{new Date(e.timestamp).toLocaleTimeString()}</span>
              <span className="event-type-tag">{e.event_type}</span>
              <span className="event-reason">{e.reason}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
