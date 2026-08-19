"""
src/observability/logging_config.py

Rotating, structured (JSON-lines) logging shared across services.

Every log record carries: timestamp, service, level, message, and an
optional correlation_id -- the correlation_id plumbing is added now so
Phase 7 (event-driven architecture) doesn't need to touch every call site
later; for Phase 1 it's simply None unless a caller passes one.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_CONFIGURED_LOGGERS: Dict[str, logging.Logger] = {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "service": getattr(record, "service", record.name),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(
    service_name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    Return a configured logger for `service_name`. Safe to call repeatedly
    (e.g. once per module) -- configuration only happens once per service
    name, subsequent calls return the same logger.
    """
    if service_name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[service_name]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    logger.propagate = False

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{service_name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)

    _CONFIGURED_LOGGERS[service_name] = logger
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    correlation_id: Optional[str] = None,
    **extra_fields: Any,
) -> None:
    """Convenience wrapper so call sites don't need to build `extra=` dicts by hand."""
    logger.log(
        level,
        message,
        extra={
            "service": logger.name,
            "correlation_id": correlation_id,
            "extra_fields": extra_fields,
        },
    )
