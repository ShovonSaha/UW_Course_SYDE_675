import numpy as np
import torch
import torchvision
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def kmeans_from_scratch(X, n_clusters, distance='cosine', max_iter=20):
    """
    Perform K-means clustering from scratch.
    
    Args:
    - X (numpy.ndarray): Input data points
    - n_clusters (int): Number of clusters
    - distance (str): Distance metric to use: 'cosine'
    - max_iter (int): Maximum number of iterations
    
    Returns:
    - labels (numpy.ndarray): Cluster labels for each data point
    - centroids (numpy.ndarray): Final centroid positions
    """
    print(f"Running kmeans_from_scratch function with distance metric: {distance}")
    
    # Initialize centroids randomly
    centroids_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[centroids_indices]
    
    # Initialize labels array
    labels = np.zeros(X.shape[0])
    
    # Iterate until convergence or max_iter
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        # Assign each data point to the nearest centroid
        for i, x in enumerate(X):
            if distance == 'cosine':
                distances = [1 - cosine_similarity([x], [centroid])[0][0] for centroid in centroids]
            else:
                raise ValueError("Invalid distance metric. Choose 'cosine'.")
                
            labels[i] = np.argmin(distances)
        
        # Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            print("Convergence reached.")
            break
        
        centroids = new_centroids
    
    print("Finished executing kmeans_from_scratch function.")
    
    return labels, centroids

# Load the MNIST dataset using torchvision
print("Loading MNIST dataset...")
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
X = mnist_train.data.numpy()
y = mnist_train.targets.numpy()

# Preprocess the data by flattening the images
print("Preprocessing the data...")
X_flattened = X.reshape(X.shape[0], -1)

# Perform K-means clustering with cosine similarity
print("Running K-means clustering with Cosine similarity...")
labels_cosine_similarity, centroids_cosine_similarity = kmeans_from_scratch(X_flattened, n_clusters=10, distance='cosine')

# Visualize cluster centers and data points
plt.figure(figsize=(10, 6))

# Plot data points colored by cluster assignment
for cluster_label in np.unique(labels_cosine_similarity):
    cluster_points = X_flattened[labels_cosine_similarity == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label}', alpha=0.5)

# Plot cluster centers
plt.scatter(centroids_cosine_similarity[:, 0], centroids_cosine_similarity[:, 1], c='red', marker='x', label='Cluster Centers', s=100)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering Results with Cosine Similarity')
plt.legend()
plt.show()

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(X_flattened, labels_cosine_similarity, metric='cosine')
print(f"Silhouette Score: {silhouette_avg}")