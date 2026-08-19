import numpy as np
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit_machine_learning.algorithms import VQC

# Load dataset
X_train = np.load("X_train_pca.npy")[:200]
y_train = np.load("y_train.npy")[:200]

X_test = np.load("X_test_pca.npy")[:100]
y_test = np.load("y_test.npy")[:100]

num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
ansatz = RealAmplitudes(num_qubits, reps=2)

optimizers = {
    "COBYLA": COBYLA(maxiter=100),
    "SPSA": SPSA(maxiter=100),
    "ADAM": ADAM(maxiter=40)
}

results = {}

for name, opt in optimizers.items():

    print(f"\nTraining VQC with {name}")

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=opt
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(name, "Accuracy:", acc)

print("\n===== Optimizer Comparison =====")

for k, v in results.items():
    print(k, ":", v)