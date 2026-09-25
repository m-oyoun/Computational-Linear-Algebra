import numpy as np
np.random.seed(42)
X_raw = np.random.randn(100, 2)
X_raw[:, 1] = X_raw[:, 0] * 2 + np.random.randn(100) * 0.3

print("--- PCA From Scratch Engine ---")
print(f"Original Data Matrix Shape: {X_raw.shape} (2 Dimensions)")
mean = np.mean(X_raw, axis=0)
X_centered = X_raw - mean
N = len(X_raw)
covariance_matrix = np.dot(X_centered.T, X_centered) / (N - 1)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

top_vector = eigenvectors[:, 0]
X_compressed = np.dot(X_centered, top_vector)

print(f"Compressed Data Matrix Shape: {X_compressed.shape} (1 Dimension!)")
print("\n--- Eigenvalue Analysis Complete ---")
print(f"Dominant Direction Vector (PC1): {top_vector}")
print(f"Variance explained by PC1: {eigenvalues[0]:.4f}")
