import numpy as np
from torchvision import datasets, transforms
from scipy.spatial.distance import cdist

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy().reshape(-1, 28*28)
y_train = mnist_train.targets.numpy()

# Choose classes 3 and 4
X_34 = X_train[(y_train == 3) | (y_train == 4)]
y_34 = y_train[(y_train == 3) | (y_train == 4)]

# Function to calculate cosine distance
def cosine_distance(X, centroids):
    return cdist(X, centroids, metric='cosine')

# Function to calculate Mahalanobis distance
def mahalanobis_distance(X, centroids, epsilon=1e-6):
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov + epsilon * np.eye(cov.shape[0]))
    return cdist(X, centroids, metric='mahalanobis', VI=inv_cov)

# Function to initialize centroids
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# Function to perform K-means clustering
def kmeans(X, k, distance_metric='euclidean', max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        if distance_metric == 'euclidean':
            distances = cdist(X, centroids)
        elif distance_metric == 'cosine':
            distances = cosine_distance(X, centroids)
        elif distance_metric == 'mahalanobis':
            distances = mahalanobis_distance(X, centroids)
        else:
            raise ValueError("Invalid distance metric")
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Function to calculate inertia
def calculate_inertia(X, centroids, labels):
    return np.sum(np.min(cdist(X, centroids, 'euclidean'), axis=1))

# Perform K-means clustering with different distance metrics
k = 2  # Number of clusters
centroids_euclidean, labels_euclidean = kmeans(X_34, k, distance_metric='euclidean')
centroids_cosine, labels_cosine = kmeans(X_34, k, distance_metric='cosine')
centroids_mahalanobis, labels_mahalanobis = kmeans(X_34, k, distance_metric='mahalanobis')

# Evaluate clustering performance using inertia
inertia_euclidean = calculate_inertia(X_34, centroids_euclidean, labels_euclidean)
inertia_cosine = calculate_inertia(X_34, centroids_cosine, labels_cosine)
inertia_mahalanobis = calculate_inertia(X_34, centroids_mahalanobis, labels_mahalanobis)

print("Inertia (Euclidean):", inertia_euclidean)
print("Inertia (Cosine):", inertia_cosine)
print("Inertia (Mahalanobis):", inertia_mahalanobis)