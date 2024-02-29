import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_distances

# Function to sample approximately n_samples per class
def sample_per_class(X, y, n_samples):
    sampled_indices = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        sampled_indices.extend(np.random.choice(indices, n_samples, replace=False))
    return X[sampled_indices], y[sampled_indices]

# Function to normalize features
def normalize_features(X):
    return X / 255.0

# Function to calculate cluster consistency
def cluster_consistency(labels, cluster_labels, num_classes):
    cluster_consistency_values = []
    for cluster_label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        class_counts = np.zeros(num_classes)
        for i in cluster_indices:
            class_counts[labels[i]] += 1
        most_common_class_count = np.max(class_counts)
        total_cluster_points = len(cluster_indices)
        cluster_consistency = most_common_class_count / total_cluster_points
        cluster_consistency_values.append(cluster_consistency)
    overall_consistency = np.mean(cluster_consistency_values)
    return overall_consistency

# Function to compute clustering accuracy
def clustering_accuracy(labels, cluster_labels, num_classes):
    correct = 0
    for cluster_label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        cluster_class_counts = np.zeros(num_classes)
        for i in cluster_indices:
            cluster_class_counts[labels[i]] += 1
        most_common_class = np.argmax(cluster_class_counts)
        correct += cluster_class_counts[most_common_class]
    accuracy = (correct / len(labels))*100
    return accuracy

# Function to perform K-means clustering with cosine distance
def kmeans_cosine(X, k, max_iter=30):
    # K-means++ initialization
    centroids = [X[np.random.choice(len(X))]]
    for _ in range(1, k):
        distances = np.array([np.min(cosine_distances(x.reshape(1, -1), centroids)) for x in X])
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
    return labels, centroids

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, centroid, cov_inv):
    diff = x - centroid
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# Using a lower number of iterations because of limited time and computation power
def kmeans_mahalanobis(X, k, max_iter=30):
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
    return labels, centroids

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
k_values = [5, 10, 20, 40, 200]

# Calculate cluster consistency and accuracy for each value of k using cosine distance
print("Results using cosine distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply K-means clustering with cosine distance
    labels_cosine, _ = kmeans_cosine(X_sampled, k)
    
    # Calculate cluster consistency
    consistency = cluster_consistency(y_sampled, labels_cosine, num_classes=10)
    print(f"    Cluster consistency: {consistency:.4f}")
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_cosine, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.2f}%")

# Calculate cluster consistency and accuracy for each value of k using Mahalanobis distance
print("\nResults using Mahalanobis distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply K-means clustering with Mahalanobis distance
    labels_mahalanobis, _ = kmeans_mahalanobis(X_sampled, k)
    
    # Calculate cluster consistency
    consistency = cluster_consistency(y_sampled, labels_mahalanobis, num_classes=10)
    print(f"    Cluster consistency: {consistency:.4f}")
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_mahalanobis, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.2f}%")