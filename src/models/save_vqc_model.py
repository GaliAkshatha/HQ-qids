import numpy as np
import pickle

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA


X_train = np.load("X_train_pca.npy")
y_train = np.load("y_train.npy")

# Reduce dataset (quantum models are slow)

X_train = X_train[:200]
y_train = y_train[:200]

print("Training data:", X_train.shape)


num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=2
)

ansatz = RealAmplitudes(
    num_qubits,
    reps=2
)

optimizer = COBYLA(maxiter=50)


model = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

print("Training VQC model...")

model.fit(X_train, y_train)

print("Training complete.")


with open("vqc_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("VQC model saved as vqc_model.pkl")