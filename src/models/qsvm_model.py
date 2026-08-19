
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel


X_train_full = np.load("X_train_pca.npy")
X_test_full = np.load("X_test_pca.npy")
y_train_full = np.load("y_train.npy")
y_test_full = np.load("y_test.npy")

print("Dataset Loaded")
print("Original Train Shape:", X_train_full.shape)
print("Original Test Shape:", X_test_full.shape)


X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=300,              # optimal size
    stratify=y_train_full,
    random_state=42
)

X_test, _, y_test, _ = train_test_split(
    X_test_full, y_test_full,
    train_size=150,
    stratify=y_test_full,
    random_state=42
)

print("\nAfter Subsampling:")
print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(
    feature_dimension=num_qubits,
    reps=2   # keep low for speed
)

print("\nFeature map created with", num_qubits, "qubits")


quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)


print("\nComputing / Loading Kernel Matrix...")

if os.path.exists("kernel_train.npy"):
    kernel_train = np.load("kernel_train.npy")
    print("Loaded cached TRAIN kernel")
else:
    kernel_train = quantum_kernel.evaluate(x_vec=X_train)
    np.save("kernel_train.npy", kernel_train)
    print("Computed and saved TRAIN kernel")

if os.path.exists("kernel_test.npy"):
    kernel_test = np.load("kernel_test.npy")
    print("Loaded cached TEST kernel")
else:
    kernel_test = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)
    np.save("kernel_test.npy", kernel_test)
    print("Computed and saved TEST kernel")


model = SVC(kernel="precomputed")

print("\nTraining QSVM...")

model.fit(kernel_train, y_train)


y_pred = model.predict(kernel_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== QSVM Results =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


qsvm_bundle = {
    "model": model,
    "quantum_kernel": quantum_kernel,
    "X_train": X_train
}

joblib.dump(qsvm_bundle, "qsvm_model_optimized.pkl")

print("\nQSVM model bundle saved successfully")