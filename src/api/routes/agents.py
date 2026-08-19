from flask import Blueprint, current_app, jsonify

agents_bp = Blueprint("agents", __name__)


@agents_bp.get("/agents")
def list_agents():
    service = current_app.config["EXPERIMENT_SERVICE"]
    summary = {}
    for experiment in service.list_experiments():
        for inc in experiment.incidents:
            key = inc["agent_type"]
            summary.setdefault(key, {"agent_type": key, "sessions": 0, "escalated": 0, "scenarios": set()})
            summary[key]["sessions"] += 1
            summary[key]["escalated"] += 1 if inc["escalated"] else 0
            summary[key]["scenarios"].add(inc["scenario"])
    for v in summary.values():
        v["scenarios"] = sorted(v["scenarios"])
    return jsonify({"agents": list(summary.values())})


@agents_bp.get("/agents/<agent_type>")
def get_agent(agent_type):
    service = current_app.config["EXPERIMENT_SERVICE"]
    sessions = []
    for experiment in service.list_experiments():
        for inc in experiment.incidents:
            if inc["agent_type"] == agent_type:
                sessions.append(inc)
    if not sessions:
        return jsonify({"error": "no data for this agent_type"}), 404
    return jsonify({"agent_type": agent_type, "sessions": sessions})
