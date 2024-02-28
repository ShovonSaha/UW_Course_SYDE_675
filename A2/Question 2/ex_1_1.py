import numpy as np
import torch
from sklearn.metrics import pairwise_distances
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

def compute_accuracy(true_labels, cluster_labels):
    """
    Compute accuracy by comparing true labels with cluster labels assigned by K-means.

    Args:
    - true_labels (numpy.ndarray): True labels of the data points (shape: [num_points])
    - cluster_labels (numpy.ndarray): Cluster labels assigned by K-means (shape: [num_points])

    Returns:
    - accuracy (float): Accuracy score
    """
    num_correct = np.sum(true_labels == cluster_labels)
    accuracy = num_correct / len(true_labels)
    return accuracy

def assign_clusters(X, centroids, distance_metric='cosine'):
    """
    Assign data points to the nearest cluster centroid based on the specified distance metric.

    Args:
    - X (numpy.ndarray): Data points (shape: [num_points, num_features])
    - centroids (numpy.ndarray): Centroid positions (shape: [num_clusters, num_features])
    - distance_metric (str): Distance metric to use ('cosine' or 'mahalanobis')

    Returns:
    - labels (numpy.ndarray): Cluster labels for each data point (shape: [num_points])
    """
    if distance_metric == 'mahalanobis':
        distances = mahalanobis_distance(X, centroids)
    else:
        distances = pairwise_distances(X, centroids, metric=distance_metric)

    labels = np.argmin(distances, axis=1)
    return labels

def mahalanobis_distance(X, centroids):
    """
    Compute Mahalanobis distance between data points and centroids.

    Args:
    - X (numpy.ndarray): Data points (shape: [num_points, num_features])
    - centroids (numpy.ndarray): Centroid positions (shape: [num_clusters, num_features])

    Returns:
    - distances (numpy.ndarray): Mahalanobis distances (shape: [num_points, num_clusters])
    """
    num_points, num_features = X.shape
    num_clusters = centroids.shape[0]

    distances = np.zeros((num_points, num_clusters))

    for b in range(num_points):
        for c in range(num_clusters):
            diff = X[b, :] - centroids[c, :]
            dist = np.sqrt(np.abs(np.sum(np.dot(diff, np.linalg.pinv(np.outer(diff, diff.T))))))
            distances[b, c] = dist

    return distances


# Fetch the MNIST dataset using torchvision
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=500, shuffle=True)

# Example usage:
iteration_count = 1
ksize = 10
num_points_per_class = 500  # Use ~500 images per class
num_features = 28 * 28  # MNIST images are 28x28 pixels
num_classes = 10  # MNIST has 10 classes (digits 0-9)

# Get data points and true labels for clustering
X = []
true_labels = []
for images, labels in train_loader:
    images = images.view(images.size(0), -1).numpy()  # Flatten images to 1D arrays
    X.append(images[:num_points_per_class])
    true_labels.append(labels[:num_points_per_class])
X = np.vstack(X)
true_labels = np.hstack(true_labels)

# Generate random initial centroids for K-means clustering
centroids = np.random.randn(ksize, num_features)

# Perform K-means clustering iterations using cosine distance
for a in range(iteration_count):
    # Assign data points to clusters based on cosine distance
    labels = assign_clusters(X, centroids, distance_metric='cosine')
    
    # Update centroids based on cluster assignments
    for c in range(ksize):
        cluster_points = X[labels == c]
        if len(cluster_points) > 0:
            centroids[c] = np.mean(cluster_points, axis=0)

# Compute accuracy for cosine distance
cosine_accuracy = compute_accuracy(true_labels, labels)

print("Accuracy using cosine distance:", cosine_accuracy)

# Perform K-means clustering iterations using Mahalanobis distance
for a in range(iteration_count):
    # Assign data points to clusters based on Mahalanobis distance
    labels = assign_clusters(X, centroids, distance_metric='mahalanobis')
    
    # Update centroids based on cluster assignments
    for c in range(ksize):
        cluster_points = X[labels == c]
        if len(cluster_points) > 0:
            centroids[c] = np.mean(cluster_points, axis=0)

# Compute accuracy for Mahalanobis distance
mahalanobis_accuracy = compute_accuracy(true_labels, labels)

print("Accuracy using Mahalanobis distance:", mahalanobis_accuracy)
