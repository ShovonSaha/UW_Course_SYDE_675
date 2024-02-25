import numpy as np
import torch
import torchvision
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score

def compute_mahalanobis_distance(X, centroid, covariance_matrix, regularization=1e-5):
    """
    Compute Mahalanobis distance between data points and a centroid using regularization.

    Args:
    - X (numpy.ndarray): Data points
    - centroid (numpy.ndarray): Centroid position
    - covariance_matrix (numpy.ndarray): Covariance matrix
    - regularization (float): Regularization parameter (default: 1e-5)

    Returns:
    - mahalanobis_distances (numpy.ndarray): Mahalanobis distances
    """
    # Regularize the covariance matrix
    covariance_matrix_regularized = covariance_matrix + regularization * np.eye(covariance_matrix.shape[0])

    try:
        # Compute the inverse of the regularized covariance matrix
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix_regularized)
    except np.linalg.LinAlgError:
        # Handle singular matrix by adding a small regularization term
        inverse_covariance_matrix = np.linalg.inv(covariance_matrix_regularized + regularization)

    # Compute the difference between data points and centroid
    centered_data = X - centroid

    # Compute Mahalanobis distances
    mahalanobis_distances = np.sqrt(np.sum(centered_data.dot(inverse_covariance_matrix) * centered_data, axis=1))

    return mahalanobis_distances

def kmeans_from_scratch(X, n_clusters, distance='cosine', max_iter=2):
    """
    Perform K-means clustering from scratch.
    
    Args:
    - X (numpy.ndarray): Input data points
    - n_clusters (int): Number of clusters
    - distance (str): Distance metric to use: 'cosine' or 'mmd'
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
                distances = cosine_distances([x], centroids)[0]
            elif distance == 'mmd':
                distances = compute_mahalanobis_distance(X, x, np.cov(X.T))
            else:
                raise ValueError("Invalid distance metric. Choose 'cosine' or 'mmd'.")
                
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

# Perform K-means clustering with cosine distance
print("Running K-means clustering with cosine distance...")
# labels_cosine, centroids_cosine = kmeans_from_scratch(X_flattened, n_clusters=5, distance='cosine', max_iter=2)

# Perform K-means clustering with Mahalanobis (MMD) distance
print("Running K-means clustering with Mahalanobis (MMD) distance...")
labels_mmd, centroids_mmd = kmeans_from_scratch(X_flattened, n_clusters=5, distance='mmd', max_iter=2)

# Evaluate clustering performance using silhouette score
# silhouette_avg_cosine = silhouette_score(X_flattened, labels_cosine)
silhouette_avg_mmd = silhouette_score(X_flattened, labels_mmd)
# print(f"Silhouette Score (Cosine Distance): {silhouette_avg_cosine}")
print(f"Silhouette Score (Mahalanobis Distance): {silhouette_avg_mmd}")
