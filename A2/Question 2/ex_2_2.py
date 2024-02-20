import numpy as np
import torchvision
from sklearn.metrics.pairwise import cosine_distances

def euclidean_distance(x1, x2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def cosine_distance(x1, x2):
    """Calculate the cosine distance between two points using scikit-learn."""
    return cosine_distances([x1], [x2])[0][0]

def mahalanobis_distance(x, y, cov_inv):
    """Calculate the Mahalanobis distance between two points."""
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

def kmeans_from_scratch(X, n_clusters, distance='euclidean', max_iter=20):
    """
    Perform K-means clustering from scratch.
    
    Args:
    - X (numpy.ndarray): Input data points
    - n_clusters (int): Number of clusters
    - distance (str): Distance metric to use: 'euclidean', 'cosine', or 'mahalanobis'
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
            if distance == 'euclidean':
                distances = [euclidean_distance(x, centroid) for centroid in centroids]
            elif distance == 'cosine':
                distances = [cosine_distance(x, centroid) for centroid in centroids]
            elif distance == 'mahalanobis':
                distances = [mahalanobis_distance(x, centroid, np.linalg.inv(np.cov(X.T))) for centroid in centroids]
            else:
                raise ValueError("Invalid distance metric. Choose 'euclidean', 'cosine', or 'mahalanobis'.")
                
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

# Perform K-means clustering with different distance metrics
print("Running K-means clustering with Euclidean distance...")
labels_euclidean, centroids_euclidean = kmeans_from_scratch(X_flattened, n_clusters=10, distance='euclidean')

print("Running K-means clustering with Cosine distance...")
labels_cosine, centroids_cosine = kmeans_from_scratch(X_flattened, n_clusters=10, distance='cosine')

print("Running K-means clustering with Mahalanobis distance...")
labels_mahalanobis, centroids_mahalanobis = kmeans_from_scratch(X_flattened, n_clusters=10, distance='mahalanobis')

# Display the results
print("\nResults:")
print("Euclidean Distance:")
print("Euclidean labels:", labels_euclidean)
print("Euclidean centroids:", centroids_euclidean)

print("\nCosine Distance:")
print("Cosine labels:", labels_cosine)
print("Cosine centroids:", centroids_cosine)

print("\nMahalanobis Distance:")
print("Mahalanobis labels:", labels_mahalanobis)
print("Mahalanobis centroids:", centroids_mahalanobis)
