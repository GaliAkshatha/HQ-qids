import numpy as np
import pickle
from sklearn.svm import SVC

# load your PCA reduced dataset
X_train = np.load("X_train_pca.npy")
y_train = np.load("y_train.npy")

print("Training SVM model...")

model = SVC(probability=True)

model.fit(X_train, y_train)

print("Training complete")

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")