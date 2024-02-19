import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Reshape the data to flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)

# Filter dataset to contain only classes 3 and 4
X_34 = X_train[(y_train == 3) | (y_train == 4)]
y_34 = y_train[(y_train == 3) | (y_train == 4)]

# Normalize data
X_34 = X_34 / 255.0

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_34)

# Separate data for classes 3 and 4 after PCA
X_class3 = X_2d[y_34 == 3]
X_class4 = X_2d[y_34 == 4]

# Calculate the mean of each class
mean_class3 = np.mean(X_class3, axis=0)
mean_class4 = np.mean(X_class4, axis=0)

# Calculate the covariance matrices for each class
cov_class3 = np.cov(X_class3.T)
cov_class4 = np.cov(X_class4.T)

# Perform eigenvalue decomposition on covariance matrices
eigvals3, eigvecs3 = np.linalg.eigh(cov_class3)
eigvals4, eigvecs4 = np.linalg.eigh(cov_class4)

# Invert the eigenvalues to get the inverse covariance matrices
cov_inv_class3 = np.dot(eigvecs3, np.dot(np.diag(1.0 / eigvals3), eigvecs3.T))
cov_inv_class4 = np.dot(eigvecs4, np.dot(np.diag(1.0 / eigvals4), eigvecs4.T))

# Calculate the slope of the MED decision boundary
slope_med = -(mean_class4[0] - mean_class3[0]) / (mean_class4[1] - mean_class3[1])

# Calculate the midpoint of the MED decision boundary
midpoint_med = [(mean_class3[0] + mean_class4[0]) / 2, (mean_class3[1] + mean_class4[1]) / 2]

# Calculate the MMD decision boundary
mean_diff = mean_class3 - mean_class4
cov_avg_inv = (cov_inv_class3 + cov_inv_class4) / 2

# Calculate the slope of the MMD decision boundary
slope_mmd = -cov_avg_inv[0, 0] / cov_avg_inv[0, 1]

# Calculate the intercept of the MMD decision boundary
intercept_mmd = np.dot(cov_avg_inv, mean_diff)[1]

# Plotting decision boundaries and training data
plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue', alpha=0.5)
plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red', alpha=0.5)
plt.scatter(mean_class3[0], mean_class3[1], marker='x', s=200, c='green', label='Class 3 Mean')
plt.scatter(mean_class4[0], mean_class4[1], marker='o', s=200, c='purple', label='Class 4 Mean')

# Plot the MED decision boundary
x_med = np.linspace(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]), 100)
y_med = slope_med * (x_med - midpoint_med[0]) + midpoint_med[1]
plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='black')

# Plot the MMD decision boundary
x_mmd = np.linspace(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]), 100)
y_mmd = slope_mmd * x_mmd + intercept_mmd
plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='green')

plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()