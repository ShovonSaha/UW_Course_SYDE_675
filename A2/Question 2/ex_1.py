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

def cosine_distance(x, centroid):
    return np.dot(x, centroid) / (np.linalg.norm(x) * np.linalg.norm(centroid))

def kmeans_cosine(X, k, max_iter=100):
    # K-means++ initialization
    centroids = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = np.array([np.min([cosine_distance(x, centroid) for centroid in centroids]) for x in X])
        prob = distances / np.sum(distances)
        centroids.append(X[np.random.choice(len(X), p=prob)])
    centroids = np.array(centroids)
    
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
    # K-means++ initialization
    centroids = [X[np.random.choice(len(X))]]
    cov_inv = np.linalg.pinv(np.cov(X.T))
    for _ in range(1, k):
        distances = np.array([np.min([mahalanobis_distance(x, centroid, cov_inv) for centroid in centroids]) for x in X])
        prob = distances / np.sum(distances)
        centroids.append(X[np.random.choice(len(X), p=prob)])
    centroids = np.array(centroids)
       
    
    for _ in range(max_iter):
        distances = np.array([mahalanobis_distance(x, centroid, cov_inv) for centroid in centroids for x in X])
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

# Preprocess data to handle NaN values
nan_indices = np.isnan(X)
X[nan_indices] = 0  # Replace NaN values with zeros
X = X[~np.any(nan_indices, axis=1)]  # Remove rows containing NaN values

# Sample approximately 500 images per class
X_sampled, y_sampled = sample_per_class(X, y, n_samples=500)

# Normalize features
X_sampled = normalize_features(X_sampled)

# Define value of k
k_values = [10]

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