"""
src/runtime/tracing.py

Reconstructs the lineage of one correlation_id across every service's
existing structured JSON log files (src/observability/logging_config.py,
Phase 1 -- unmodified). No new logging infrastructure: every worker
already logs correlation_id/causation_id/incident_id/event_type/latency
as extra_fields (logging_config.py already flattens arbitrary
extra_fields into the JSON payload, so this needed zero changes there).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"


@dataclass
class TraceEntry:
    timestamp: str
    service: str
    event_type: Optional[str]
    causation_id: Optional[str]
    incident_id: Optional[str]
    message: str
    level: str
    raw: Dict[str, Any]


def trace_correlation(correlation_id: str, log_dir: str | Path = DEFAULT_LOG_DIR) -> List[TraceEntry]:
    """
    Scans every *.log file in log_dir, parses each JSON line, filters to
    entries matching correlation_id, and returns them ordered by
    timestamp -- the full cross-service lineage for one traffic sample's
    journey through the pipeline.
    """
    log_dir = Path(log_dir)
    entries: List[TraceEntry] = []

    if not log_dir.exists():
        return entries

    for log_file in sorted(log_dir.glob("*.log")):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("correlation_id") != correlation_id:
                    continue
                entries.append(
                    TraceEntry(
                        timestamp=record.get("timestamp", ""),
                        service=record.get("service", log_file.stem),
                        event_type=record.get("event_type"),
                        causation_id=record.get("causation_id"),
                        incident_id=record.get("incident_id"),
                        message=record.get("message", ""),
                        level=record.get("level", ""),
                        raw=record,
                    )
                )

    entries.sort(key=lambda e: e.timestamp)
    return entries


def format_trace(entries: List[TraceEntry]) -> str:
    if not entries:
        return "(no trace entries found)"
    lines = []
    for e in entries:
        lines.append(f"{e.timestamp}  [{e.service:20s}]  {e.event_type or '-':30s}  {e.message}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m src.runtime.tracing <correlation_id>")
        sys.exit(1)
    result = trace_correlation(sys.argv[1])
    print(format_trace(result))
