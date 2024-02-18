import torch
import torchvision
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# Define transform to flatten images
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.flatten())
])

# Load MNIST training dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Extract data and labels
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Reshape and normalize data
X_normalized = X_train.reshape(X_train.shape[0], -1) / 255.0

# Define number of clusters
n_clusters = 10

# Initialize K-means with different distance metrics
kmeans_euclidean = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_cosine = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

# Fit K-means models
kmeans_euclidean.fit(X_normalized)
kmeans_cosine.fit(X_normalized)

# Calculate cosine distance matrix
cosine_distance_matrix = cosine_distances(X_normalized)

# # Evaluate clustering results
# inertia_euclidean = kmeans_euclidean.inertia_
# silhouette_euclidean = silhouette_score(X_normalized, kmeans_euclidean.labels_)
# inertia_cosine = kmeans_cosine.inertia_
# silhouette_cosine = silhouette_score(X_normalized, kmeans_cosine.labels_)
# print("Euclidean Distance - Inertia:", inertia_euclidean)
# print("Euclidean Distance - Silhouette Score:", silhouette_euclidean)
# print("Cosine Distance - Inertia:", inertia_cosine)
# print("Cosine Distance - Silhouette Score:", silhouette_cosine)

# Plot clustering results or visualize cluster centroids
# For visualization, you may need to reshape centroids to 28x28 if you want to display them as images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot centroids for Euclidean distance
axes[0].set_title('K-means Clustering - Euclidean Distance')
for i in range(n_clusters):
    centroid_image = kmeans_euclidean.cluster_centers_[i].reshape(28, 28)
    axes[0].imshow(centroid_image, cmap='gray')
    axes[0].axis('off')
plt.tight_layout()

# Plot centroids for Cosine distance
axes[1].set_title('K-means Clustering - Cosine Distance')
for i in range(n_clusters):
    centroid_index = np.argmin(np.sum(cosine_distance_matrix[:, kmeans_cosine.labels_ == i], axis=1))
    centroid_image = X_normalized[centroid_index].reshape(28, 28)
    axes[1].imshow(centroid_image, cmap='gray')
    axes[1].axis('off')
plt.tight_layout()
plt.show()

# # Compare clustering results obtained with different distance metrics
# print("Comparison of Clustering Results:")
# print("Euclidean Distance - Inertia:", inertia_euclidean)
# print("Cosine Distance - Inertia:", inertia_cosine)
# print("Euclidean Distance - Silhouette Score:", silhouette_euclidean)
# print("Cosine Distance - Silhouette Score:", silhouette_cosine)