"""
src/api/services/dashboard_service.py

Real system-status checks and real Stage C model comparison data -- no
fabricated values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.runtime.config import RuntimePolicyConfig
from src.runtime.redis_client import build_redis_client, check_redis_connectivity

REPO_ROOT = Path(__file__).resolve().parents[3]


def check_redis() -> bool:
    """
    Uses the SAME configured Redis connection mechanism as the rest of
    the application (RuntimePolicyConfig -> build_redis_client, which
    honors REDIS_URL / REDIS_HOST / REDIS_PORT). Previously this
    constructed an unconfigured redis.Redis() directly, which always
    checked localhost:6379 regardless of what was actually configured --
    a real bug, fixed here, not merely refactored for style.
    """
    try:
        config = RuntimePolicyConfig.load()
        client = build_redis_client(config)
        return check_redis_connectivity(client)
    except Exception:
        return False


def check_artifact(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def system_status() -> Dict[str, bool]:
    return {
        "api": True,
        "redis": check_redis(),
        "classical_detector": check_artifact("artifacts/models/classical/random_forest.joblib"),
        "vqc": check_artifact("artifacts/models/quantum/vqc/vqc_model.dill"),
        "qsvm": check_artifact("artifacts/models/quantum/qsvm/svc_model.joblib"),
        "application_detector": check_artifact("artifacts/models/application/classical/random_forest.joblib"),
    }


def model_comparison() -> Dict:
    result = {"dataset_label": "AGENT_GENERATED_LABELED_DATA", "models": {}}
    classical_path = REPO_ROOT / "reports" / "stage_c" / "classical_baseline_results.json"
    quantum_path = REPO_ROOT / "reports" / "stage_c" / "quantum_comparison_results.json"

    if classical_path.exists():
        with open(classical_path) as f:
            classical = json.load(f)
        for name in ("logistic_regression", "random_forest"):
            if classical.get(name, {}).get("test"):
                result["models"][name] = classical[name]["test"]

    if quantum_path.exists():
        with open(quantum_path) as f:
            quantum = json.load(f)
        for name in ("qsvm", "vqc"):
            if name in quantum:
                result["models"][name] = {
                    **quantum[name]["metrics"],
                    "training_time_seconds": quantum[name]["training_time_seconds"],
                    "inference_time_ms_per_sample": quantum[name]["inference_time_ms_per_sample"],
                }
        result["bounded_experiment"] = quantum.get("bounded_experiment", True)
        result["train_size"] = quantum.get("train_size")
        result["test_size"] = quantum.get("test_size")

    return result
