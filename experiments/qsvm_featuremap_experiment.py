import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Load dataset
X_train = np.load("X_train_pca.npy")[:300]
y_train = np.load("y_train.npy")[:300]

X_test = np.load("X_test_pca.npy")[:150]
y_test = np.load("y_test.npy")[:150]

num_qubits = X_train.shape[1]

# Define feature maps
feature_maps = {
    "ZZFeatureMap": ZZFeatureMap(feature_dimension=num_qubits, reps=2),
    "ZFeatureMap": ZFeatureMap(feature_dimension=num_qubits, reps=2),
    "PauliFeatureMap": PauliFeatureMap(feature_dimension=num_qubits, reps=2)
}

results = {}

for name, fmap in feature_maps.items():

    print(f"\nRunning QSVM with {name}")

    kernel = FidelityQuantumKernel(feature_map=fmap)

    kernel_train = kernel.evaluate(x_vec=X_train)
    kernel_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)

    model = SVC(kernel="precomputed")
    model.fit(kernel_train, y_train)

    y_pred = model.predict(kernel_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Accuracy:", acc)

print("\n===== Feature Map Comparison =====")

for k, v in results.items():
    print(k, ":", v)