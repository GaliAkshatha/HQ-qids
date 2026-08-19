"""
src/agents/application_quantum.py

Bounded quantum comparison for the application-security detector. Reuses
the EXISTING quantum architecture (build_feature_map, FidelityQuantumKernel,
QSVMVerifier, VQCVerifier) with NEW artifacts fit on the 11-dim
application feature space -- never the NSL-KDD-trained objects.

HONESTY NOTE: the AGENT_GENERATED_LABELED_DATA dataset here is small
(tens of sessions, not thousands). This is a BOUNDED, exploratory
comparison, not a scientifically powered experiment. Any accuracy
difference between classical/quantum on a dataset this size should not
be read as evidence that quantum verification helps application-security
detection in general -- it is reported descriptively, not as a
significance claim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from src.agents.application_dataset import LabeledSample, to_matrix
from src.agents.application_detector import evaluate
from src.quantum.feature_maps import build_feature_map
from src.quantum.qsvm_verifier import QSVMVerifier
from src.quantum.vqc_verifier import VQCVerifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUANTUM_DIR = REPO_ROOT / "artifacts" / "models" / "application" / "quantum"
DEFAULT_PREPROCESSING_DIR = REPO_ROOT / "artifacts" / "preprocessing" / "application"

N_QUBITS = 2


def train_application_quantum(
    train: List[LabeledSample], test: List[LabeledSample], scaler,
    quantum_dir: Path = DEFAULT_QUANTUM_DIR, preprocessing_dir: Path = DEFAULT_PREPROCESSING_DIR,
) -> Dict[str, Any]:
    quantum_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = to_matrix(train)
    X_test, y_test = to_matrix(test)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=N_QUBITS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    joblib.dump(pca, preprocessing_dir / "quantum_pca.joblib")

    feature_map_config = {"type": "ZZFeatureMap", "num_qubits": N_QUBITS, "reps": 2}
    feature_map = build_feature_map(feature_map_config)

    results: Dict[str, Any] = {"n_qubits": N_QUBITS, "train_size": len(train), "test_size": len(test),
                                "dataset_label": "AGENT_GENERATED_LABELED_DATA", "bounded_experiment": True}

    qsvm_dir = quantum_dir / "qsvm"
    qsvm_dir.mkdir(parents=True, exist_ok=True)
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    t0 = time.perf_counter()
    K_train = kernel.evaluate(x_vec=X_train_pca)
    svc = SVC(kernel="precomputed", probability=True, random_state=42)
    svc.fit(K_train, y_train)
    qsvm_train_time = time.perf_counter() - t0

    joblib.dump(svc, qsvm_dir / "svc_model.joblib")
    np.save(qsvm_dir / "reference_vectors.npy", X_train_pca)
    with open(qsvm_dir / "feature_map_config.json", "w") as f:
        json.dump(feature_map_config, f)

    qsvm_verifier = QSVMVerifier(svc_model=svc, reference_vectors=X_train_pca, feature_map_config=feature_map_config, quantum_pca=pca)
    t0 = time.perf_counter()
    qsvm_results = qsvm_verifier.verify_batch(X_test_scaled, [s.session_id for s in test])
    qsvm_inference_time = time.perf_counter() - t0
    qsvm_pred = [0 if r.quantum_prediction == "normal" else 1 for r in qsvm_results]

    results["qsvm"] = {
        "metrics": evaluate(y_test, qsvm_pred),
        "training_time_seconds": qsvm_train_time,
        "inference_time_seconds_total": qsvm_inference_time,
        "inference_time_ms_per_sample": (qsvm_inference_time / max(len(test), 1)) * 1000,
    }

    vqc_dir = quantum_dir / "vqc"
    vqc_dir.mkdir(parents=True, exist_ok=True)
    ansatz_config = {"type": "RealAmplitudes", "reps": 2}
    optimizer_config = {"type": "COBYLA", "maxiter": 50}
    ansatz = RealAmplitudes(N_QUBITS, reps=2)
    optimizer = COBYLA(maxiter=50)

    t0 = time.perf_counter()
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    vqc.fit(X_train_pca, np.array(y_train))
    vqc_train_time = time.perf_counter() - t0

    vqc.to_dill(str(vqc_dir / "vqc_model.dill"))
    with open(vqc_dir / "feature_map_config.json", "w") as f:
        json.dump(feature_map_config, f)
    with open(vqc_dir / "ansatz_config.json", "w") as f:
        json.dump(ansatz_config, f)
    with open(vqc_dir / "optimizer_config.json", "w") as f:
        json.dump(optimizer_config, f)

    vqc_verifier = VQCVerifier(vqc_model=vqc, circuit_metadata={"feature_map": feature_map_config, "ansatz": ansatz_config, "optimizer": optimizer_config}, quantum_pca=pca)
    t0 = time.perf_counter()
    vqc_results = vqc_verifier.verify_batch(X_test_scaled, [s.session_id for s in test])
    vqc_inference_time = time.perf_counter() - t0
    vqc_pred = [0 if r.quantum_prediction == "normal" else 1 for r in vqc_results]

    results["vqc"] = {
        "metrics": evaluate(y_test, vqc_pred),
        "training_time_seconds": vqc_train_time,
        "inference_time_seconds_total": vqc_inference_time,
        "inference_time_ms_per_sample": (vqc_inference_time / max(len(test), 1)) * 1000,
    }

    return results
