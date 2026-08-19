import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


X_train = np.load("X_train_pca.npy")
X_test = np.load("X_test_pca.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Dataset Loaded Successfully")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


model = SVC(kernel='rbf')

print("\nTraining Classical SVM...")

model.fit(X_train, y_train)

print("Training Completed")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Classical SVM Results =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy :", test_acc)

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

np.save("svm_predictions.npy", y_pred)