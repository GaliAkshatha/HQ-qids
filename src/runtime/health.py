"""
src/runtime/health.py

Minimal /health (liveness) and /ready (readiness) HTTP server, shared by
every worker. Readiness checks are cheap by design -- Redis PING and
on-disk artifact existence checks only, never a real inference call.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict

from flask import Flask, jsonify


def build_health_app(service_name: str, readiness_checks: Dict[str, Callable[[], bool]]) -> Flask:
    app = Flask(service_name)
    # Silence Flask's default request logging -- the structured JSON
    # logger (src/observability/logging_config.py) is this project's log
    # channel, not Flask's werkzeug access log.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/health")
    def health():
        return jsonify({"status": "alive", "service": service_name}), 200

    @app.route("/ready")
    def ready():
        results = {name: check() for name, check in readiness_checks.items()}
        all_ready = all(results.values())
        return jsonify({"status": "ready" if all_ready else "not_ready", "service": service_name, "checks": results}), (200 if all_ready else 503)

    return app


def run_health_server_in_background(app: Flask, port: int) -> threading.Thread:
    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
