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

# Function to initialize centroids using K-means++ algorithm
def kmeans_plusplus_init(X, k):
    centroids = [X[np.random.choice(len(X))]]  # Select the first centroid randomly
    for _ in range(1, k):
        distances = np.array([np.min([np.linalg.norm(x - centroid) for centroid in centroids]) for x in X])
        prob = distances / np.sum(distances)
        next_centroid_index = np.random.choice(len(X), p=prob)
        centroids.append(X[next_centroid_index])
    return np.array(centroids)

# Function to perform K-means clustering with cosine distance
def kmeans_cosine(X, k, max_iter=30):
    # Initialize centroids using K-means++
    centroids = kmeans_plusplus_init(X, k)
    
    for _ in range(max_iter):
        distances = cosine_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, centroid, cov_inv):
    diff = x - centroid
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# Function to perform K-means clustering with Mahalanobis distance
def kmeans_mahalanobis(X, k, max_iter=30):
    # Initialize centroids using K-means++
    centroids = kmeans_plusplus_init(X, k)
    
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
k_values = [5, 10, 20, 40, 200]

# Calculate clustering accuracy for each value of k using cosine distance
print("Results using cosine distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply K-means clustering with cosine distance
    labels_cosine = kmeans_cosine(X_sampled, k)
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_cosine, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.4f}")

# Calculate clustering accuracy for each value of k using Mahalanobis distance
print("\nResults using Mahalanobis distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply K-means clustering with Mahalanobis distance
    labels_mahalanobis = kmeans_mahalanobis(X_sampled, k)
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_mahalanobis, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.4f}")