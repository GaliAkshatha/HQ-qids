from flask import Blueprint, current_app, jsonify

from src.api.services import dashboard_service

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.get("/metrics")
def metrics():
    service = current_app.config["EXPERIMENT_SERVICE"]
    return jsonify({
        "system_status": dashboard_service.system_status(),
        "pipeline_metrics": service.metrics_snapshot(),
        "model_comparison": dashboard_service.model_comparison(),
    })
