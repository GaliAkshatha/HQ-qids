from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class PipelineMessage:
    """
    Wire format for Phase 7's Redis Streams. Distinct from IncidentEvent
    (Phase 6's incident audit record) -- this is inter-service transport,
    carrying whichever domain contract's .to_dict() is relevant for that
    stage (DetectionResult, RoutingDecision, HybridDecision+RiskAssessment,
    DefenseResult) in `payload`.

    event_id: unique per Redis Stream message.
    correlation_id: groups every message belonging to one traffic
        sample's journey across all 7 streams -- derived via the same
        SampleIdCorrelation logic as Phase 6 (reused, not reimplemented).
    causation_id: the event_id of the message that directly triggered
        this one (one-hop parent). None only for the root traffic.ingested
        message.
    incident_id: the stable business identity for this sample's incident,
        assigned once by the traffic gateway, carried unchanged through
        every subsequent message.
    retry_count: incremented by a worker's own retry-with-backoff logic
        (src/runtime/stream_worker.py) -- distinct from Redis's own
        pending-entry redelivery count, which XAUTOCLAIM tracks separately.
    """

    event_id: str
    correlation_id: str
    causation_id: Optional[str]
    incident_id: str
    event_type: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    schema_version: str = "1.0"

    def __post_init__(self):
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.correlation_id:
            raise ValueError("correlation_id is required")
        if not self.incident_id:
            raise ValueError("incident_id is required")
        if not self.event_type:
            raise ValueError("event_type is required")
        if self.retry_count < 0:
            raise ValueError(f"retry_count must be >= 0. Received: {self.retry_count}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineMessage":
        return cls(**data)

    def next_retry(self) -> "PipelineMessage":
        """Returns a copy with retry_count incremented -- used when a
        worker republishes after a handled failure."""
        import dataclasses

        return dataclasses.replace(self, retry_count=self.retry_count + 1)
