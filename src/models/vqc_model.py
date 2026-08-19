import numpy as np

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA

from qiskit_machine_learning.algorithms import VQC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


X_train = np.load("X_train_pca.npy")
X_test = np.load("X_test_pca.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Dataset Loaded")
print("Original Train Shape:", X_train.shape)
print("Original Test Shape:", X_test.shape)


X_train = X_train[:200]
y_train = y_train[:200]

X_test = X_test[:100]
y_test = y_test[:100]

print("Reduced Train Shape:", X_train.shape)
print("Reduced Test Shape:", X_test.shape)

num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)

ansatz = RealAmplitudes(num_qubits, reps=2)

optimizer = COBYLA(maxiter=100)


vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)


print("\nTraining VQC...")

vqc.fit(X_train, y_train)


y_pred = vqc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== VQC Results =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

np.save("vqc_predictions.npy", y_pred)