import { useEffect, useRef, useState } from "react";
import { api } from "../services/api";

export interface StreamEvent {
  event_type: string;
  raw_event_type: string;
  incident_id: string;
  correlation_id: string;
  reason: string;
  timestamp: string;
}

export type ConnectionState = "connecting" | "connected" | "disconnected";

export function useEventStream(maxEvents = 60) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let source: EventSource;
    try {
      source = api.eventSource();
    } catch {
      setConnectionState("disconnected");
      return;
    }
    sourceRef.current = source;

    source.onopen = () => setConnectionState("connected");
    source.onmessage = (msg) => {
      setConnectionState("connected");
      try {
        const data = JSON.parse(msg.data) as StreamEvent;
        setEvents((prev) => [data, ...prev].slice(0, maxEvents));
      } catch {
        // malformed frame -- skip, don't crash the stream
      }
    };
    source.onerror = () => {
      if (source.readyState === EventSource.CONNECTING) {
        setConnectionState("connecting");
      } else if (source.readyState === EventSource.CLOSED) {
        setConnectionState("disconnected");
      }
    };

    return () => source.close();
  }, [maxEvents]);

  return { events, connectionState };
}
