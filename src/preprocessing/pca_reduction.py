import numpy as np
from sklearn.decomposition import PCA


# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Data Loaded")
print("Original feature count:", X_train.shape[1])


# Apply PCA
pca = PCA(n_components=4)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


print("\nPCA Completed")
print("Reduced feature count:", X_train_pca.shape[1])


# Save reduced dataset
np.save("X_train_pca.npy", X_train_pca)
np.save("X_test_pca.npy", X_test_pca)

print("\nReduced dataset saved successfully")