import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, mahalanobis
from sklearn.covariance import EmpiricalCovariance
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define data transformation
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset using torchvision
mnist_train = MNIST(root='./data', train=True, download=True, transform=data_transform)
X = mnist_train.data.numpy()
y = mnist_train.targets.numpy()

# Flatten the data and normalize
X = X.reshape(X.shape[0], -1)
X = normalize(X)

# Define the range of clusters
k_values = [5, 10, 20, 40, 200]

# Define distance metrics
distance_metrics = ['euclidean', 'cosine', 'mahalanobis']

# Initialize lists to store evaluation metrics
inertia_scores = {metric: [] for metric in distance_metrics}
silhouette_scores = {metric: [] for metric in distance_metrics}

# Iterate over distance metrics
for metric in distance_metrics:
    print(f'Using {metric} distance metric:')
    for k in k_values:
        # Initialize k-means with the current distance metric
        if metric == 'mahalanobis':
            # Calculate the empirical covariance matrix
            cov_estimator = EmpiricalCovariance()
            cov_estimator.fit(X)
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.inertia_ = None  # Reset inertia to None to prevent using it
            kmeans._distance_func = lambda x, y: mahalanobis(x, y, cov_estimator.get_precision())
        elif metric == 'cosine':
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.inertia_ = None  # Reset inertia to None to prevent using it
            kmeans._distance_func = lambda x, y: 1 - cosine(x, y)
        else:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)

        # Fit k-means to the data
        kmeans.fit(X)

        # Calculate inertia
        inertia_scores[metric].append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette_scores[metric].append(silhouette_score(X, kmeans.labels_))

        print(f'  k={k}, Inertia: {kmeans.inertia_}, Silhouette Score: {silhouette_scores[metric][-1]}')

# Plot inertia scores for each distance metric
plt.figure(figsize=(10, 6))
for metric, scores in inertia_scores.items():
    plt.plot(k_values, scores, marker='o', label=metric)
plt.title('Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

# Plot silhouette scores for each distance metric
plt.figure(figsize=(10, 6))
for metric, scores in silhouette_scores.items():
    plt.plot(k_values, scores, marker='o', label=metric)
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()