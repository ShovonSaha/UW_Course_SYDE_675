import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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

# Function to calculate the within-cluster sum of squares (WCSS)
def calculate_wcss(X, k_values):
    wcss = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return wcss

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
k_values = range(2, 21)  # Range of k values to evaluate

# Calculate WCSS for each value of k
wcss_values = calculate_wcss(X_sampled, k_values)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss_values, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_values)
plt.grid(True)
plt.show()