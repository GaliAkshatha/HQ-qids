"""
src/quantum/feature_maps.py

Small config-driven builders shared by both the QSVM and VQC verifiers/
training scripts, so "how do we build a ZZFeatureMap" exists in exactly
one place instead of four.

Configs are plain JSON-serializable dicts so they can be persisted
alongside a trained model and used to reconstruct the exact same circuit
at load time, without pickling Qiskit circuit objects themselves.
"""

from __future__ import annotations

from typing import Any, Dict

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes, ZFeatureMap, ZZFeatureMap
from qiskit_algorithms.optimizers import ADAM, COBYLA, SPSA, Optimizer

_FEATURE_MAPS = {
    "ZZFeatureMap": ZZFeatureMap,
    "ZFeatureMap": ZFeatureMap,
    "PauliFeatureMap": PauliFeatureMap,
}

_OPTIMIZERS = {
    "COBYLA": COBYLA,
    "SPSA": SPSA,
    "ADAM": ADAM,
}


def build_feature_map(config: Dict[str, Any]) -> QuantumCircuit:
    """config: {"type": "ZZFeatureMap", "num_qubits": 4, "reps": 2}"""
    fmap_type = config["type"]
    if fmap_type not in _FEATURE_MAPS:
        raise ValueError(f"Unknown feature map type '{fmap_type}'. Known: {list(_FEATURE_MAPS)}")
    cls = _FEATURE_MAPS[fmap_type]
    return cls(feature_dimension=config["num_qubits"], reps=config.get("reps", 2))


def build_ansatz(config: Dict[str, Any]) -> QuantumCircuit:
    """config: {"type": "RealAmplitudes", "num_qubits": 4, "reps": 2}"""
    ansatz_type = config["type"]
    if ansatz_type != "RealAmplitudes":
        raise ValueError(f"Unknown ansatz type '{ansatz_type}'. Only 'RealAmplitudes' is supported currently.")
    return RealAmplitudes(config["num_qubits"], reps=config.get("reps", 2))


def build_optimizer(config: Dict[str, Any]) -> Optimizer:
    """config: {"type": "COBYLA", "maxiter": 100}"""
    opt_type = config["type"]
    if opt_type not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer type '{opt_type}'. Known: {list(_OPTIMIZERS)}")
    cls = _OPTIMIZERS[opt_type]
    return cls(maxiter=config.get("maxiter", 100))
