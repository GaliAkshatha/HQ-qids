from flask import Blueprint, jsonify

from src.api.services import dashboard_service

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    return jsonify({"status": "alive"})


@health_bp.get("/ready")
def ready():
    status = dashboard_service.system_status()
    all_ready = status["redis"] and status["classical_detector"]
    return jsonify({"status": "ready" if all_ready else "not_ready", "checks": status}), (200 if all_ready else 503)
