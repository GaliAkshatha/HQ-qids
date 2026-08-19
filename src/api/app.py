"""
src/api/app.py

Thin Flask app factory. All business logic lives in
src/api/services/*.py (which reuse existing Phase 1-8/Stage B/C domain
components) -- routes only translate HTTP <-> service calls.
"""

from __future__ import annotations

import os

from flask import Flask, request

from src.api.routes.agents import agents_bp
from src.api.routes.events import events_bp
from src.api.routes.experiments import experiments_bp
from src.api.routes.health import health_bp
from src.api.routes.incidents import incidents_bp
from src.api.routes.metrics import metrics_bp
from src.api.services.experiment_service import ExperimentService


def _allowed_origins() -> list[str] | None:
    """
    CORS_ALLOWED_ORIGINS: comma-separated list of allowed frontend
    origins (e.g. "https://qids.vercel.app,https://qids-staging.vercel.app").
    Unset -> None, meaning "allow any origin" (today's existing
    wildcard behavior, unchanged for local dev / anyone who hasn't
    configured this yet).
    """
    raw = os.environ.get("CORS_ALLOWED_ORIGINS")
    if not raw:
        return None
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> Flask:
    app = Flask(__name__)
    allowed_origins = _allowed_origins()

    @app.after_request
    def add_cors_headers(response):
        if allowed_origins is None:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            origin = request.headers.get("Origin")
            if origin in allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    app.config["EXPERIMENT_SERVICE"] = ExperimentService()

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(experiments_bp, url_prefix="/api")
    app.register_blueprint(agents_bp, url_prefix="/api")
    app.register_blueprint(incidents_bp, url_prefix="/api")
    app.register_blueprint(events_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    create_app().run(host="0.0.0.0", port=port, debug=False, threaded=True)
