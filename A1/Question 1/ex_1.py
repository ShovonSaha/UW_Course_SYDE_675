# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from torchvision import datasets, transforms

# # Load MNIST training dataset
# mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# # Convert MNIST training dataset to numpy arrays
# X_train = mnist_train.data.numpy()
# y_train = mnist_train.targets.numpy()

# # Reshape the data to flatten the images
# X_train = X_train.reshape(X_train.shape[0], -1)

# # Filter dataset to contain only classes 3 and 4
# X_34 = X_train[(y_train == 3) | (y_train == 4)]
# y_34 = y_train[(y_train == 3) | (y_train == 4)]

# # Normalize data
# X_34 = X_34 / 255.0

# # Apply PCA to reduce dimensions to 2
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X_34)

# # Separate data for classes 3 and 4 after PCA
# X_class3 = X_2d[y_34 == 3]
# X_class4 = X_2d[y_34 == 4]

# # Calculate the mean of each class
# mean_class3 = np.mean(X_class3, axis=0)
# mean_class4 = np.mean(X_class4, axis=0)

# # Calculate the covariance matrices for each class
# cov_class3 = np.cov(X_class3.T)
# cov_class4 = np.cov(X_class4.T)

# # Perform eigenvalue decomposition on covariance matrices
# eigvals3, eigvecs3 = np.linalg.eigh(cov_class3)
# eigvals4, eigvecs4 = np.linalg.eigh(cov_class4)

# # Invert the eigenvalues to get the inverse covariance matrices
# cov_inv_class3 = np.dot(eigvecs3, np.dot(np.diag(1.0 / eigvals3), eigvecs3.T))
# cov_inv_class4 = np.dot(eigvecs4, np.dot(np.diag(1.0 / eigvals4), eigvecs4.T))

# # Calculate the slope of the MED decision boundary
# slope_med = -(mean_class4[0] - mean_class3[0]) / (mean_class4[1] - mean_class3[1])

# # Calculate the midpoint of the MED decision boundary
# midpoint_med = [(mean_class3[0] + mean_class4[0]) / 2, (mean_class3[1] + mean_class4[1]) / 2]

# # Calculate the MMD decision boundary
# mean_diff = mean_class3 - mean_class4
# cov_avg_inv = (cov_inv_class3 + cov_inv_class4) / 2

# # Calculate the slope of the MMD decision boundary
# slope_mmd = -cov_avg_inv[0, 0] / cov_avg_inv[0, 1]

# # Calculate the intercept of the MMD decision boundary
# intercept_mmd = np.dot(cov_avg_inv, mean_diff)[1]

# # Plotting decision boundaries and training data
# plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue', alpha=0.5)
# plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red', alpha=0.5)
# plt.scatter(mean_class3[0], mean_class3[1], marker='x', s=200, c='green', label='Class 3 Mean')
# plt.scatter(mean_class4[0], mean_class4[1], marker='o', s=200, c='purple', label='Class 4 Mean')

# # Plot the MED decision boundary
# x_med = np.linspace(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]), 100)
# y_med = slope_med * (x_med - midpoint_med[0]) + midpoint_med[1]
# plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='black')

# # Plot the MMD decision boundary
# x_mmd = np.linspace(np.min(X_2d[:, 0]), np.max(X_2d[:, 0]), 100)
# y_mmd = slope_mmd * x_mmd + intercept_mmd
# plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='green')

# plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()



import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_distances

def sample_per_class(X, y, n_samples):
    sampled_indices = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        sampled_indices.extend(np.random.choice(indices, n_samples, replace=False))
    return X[sampled_indices], y[sampled_indices]

def normalize_features(X):
    return X / 255.0

def kmeans_cosine(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        distances = cosine_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

def mahalanobis_distance(x, centroid, cov_inv):
    diff = x - centroid
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

def kmeans_mahalanobis(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        cov_inv = np.linalg.pinv(np.cov(X.T))
        distances = np.array([mahalanobis_distance(x, centroid, cov_inv) for x in X for centroid in centroids])
        distances = distances.reshape((len(X), k))
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

# Fetch MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
X, y = mnist_trainset.data.numpy(), mnist_trainset.targets.numpy()
X = X.reshape(X.shape[0], -1)

# Sample approximately 500 images per class
X_sampled, y_sampled = sample_per_class(X, y, n_samples=500)

# Normalize features
X_sampled = normalize_features(X_sampled)

# Define values of k
k_values = [10]
# k_values = [5, 10, 20, 40, 200]

best_accuracy_cosine = 0
best_accuracy_mahalanobis = 0

# Apply K-means clustering with cosine distance for each value of k
for k in k_values:
    labels_cosine = kmeans_cosine(X_sampled, k)
    accuracy = np.mean(labels_cosine == y_sampled) * 100
    if accuracy > best_accuracy_cosine:
        best_accuracy_cosine = accuracy
    print(f"K-means clustering with cosine distance for k={k}, accuracy={accuracy:.2f}%")

# Apply K-means clustering with Mahalanobis distance for each value of k
for k in k_values:
    labels_mahalanobis = kmeans_mahalanobis(X_sampled, k)
    accuracy = np.mean(labels_mahalanobis == y_sampled) * 100
    if accuracy > best_accuracy_mahalanobis:
        best_accuracy_mahalanobis = accuracy
    print(f"K-means clustering with Mahalanobis distance for k={k}, accuracy={accuracy:.2f}%")

print(f"Best accuracy for cosine distance: {best_accuracy_cosine:.2f}%")
print(f"Best accuracy for Mahalanobis distance: {best_accuracy_mahalanobis:.2f}%")