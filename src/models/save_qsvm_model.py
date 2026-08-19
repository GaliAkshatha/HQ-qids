import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


X_train = np.load("X_train_pca.npy")
X_test = np.load("X_test_pca.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Dataset loaded")

# Reduce dataset for quantum kernel speed
X_train = X_train[:300]
y_train = y_train[:300]

X_test = X_test[:150]
y_test = y_test[:150]


num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=2
)

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)


print("Computing Quantum Kernel...")

kernel_train = quantum_kernel.evaluate(x_vec=X_train)
kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)


model = SVC(kernel="precomputed")

print("Training QSVM...")

model.fit(kernel_train, y_train)


pred = model.predict(kernel_test)

accuracy = accuracy_score(y_test, pred)

print("QSVM Accuracy:", accuracy)


joblib.dump(model, "qsvm_model.pkl")

print("QSVM model saved as qsvm_model.pkl")