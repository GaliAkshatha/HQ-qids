from flask import Blueprint, current_app, jsonify, request

experiments_bp = Blueprint("experiments", __name__)


@experiments_bp.post("/experiments/start")
def start_experiment():
    body = request.get_json(silent=True) or {}
    service = current_app.config["EXPERIMENT_SERVICE"]
    try:
        experiment = service.create_experiment(
            scenario=body.get("scenario"), n_sessions=int(body.get("n_sessions", 5)),
            mode=body.get("mode", "normal"), quantum=body.get("quantum", "auto"),
        )
        service.run_experiment(experiment.experiment_id)
        return jsonify(experiment.to_dict()), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@experiments_bp.post("/experiments/stop")
def stop_experiment():
    body = request.get_json(silent=True) or {}
    experiment_id = body.get("experiment_id")
    service = current_app.config["EXPERIMENT_SERVICE"]
    experiment = service.get_experiment(experiment_id)
    if experiment is None:
        return jsonify({"error": f"unknown experiment_id '{experiment_id}'"}), 404
    return jsonify(experiment.to_dict())


@experiments_bp.get("/experiments")
def list_experiments():
    service = current_app.config["EXPERIMENT_SERVICE"]
    return jsonify({"experiments": [e.to_dict() for e in service.list_experiments()], "scenarios": service.list_scenarios()})


@experiments_bp.get("/experiments/<experiment_id>")
def get_experiment(experiment_id):
    service = current_app.config["EXPERIMENT_SERVICE"]
    experiment = service.get_experiment(experiment_id)
    if experiment is None:
        return jsonify({"error": "not found"}), 404
    result = experiment.to_dict()
    result["incidents"] = experiment.incidents
    return jsonify(result)
