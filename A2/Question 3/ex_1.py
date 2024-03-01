# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.pairwise import pairwise_distances
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor

# # Function to calculate Mahalanobis distance
# def mahalanobis_distance(x, centroid, cov_inv):
#     diff = x - centroid
#     return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# # Using a lower number of iterations because of limited time and computation power
# def kmeans_mahalanobis(X, k, max_iter=30):
#     centroids = X[np.random.choice(len(X), k, replace=False)]
#     for _ in range(max_iter):
#         cov_inv = np.linalg.pinv(np.cov(X.T))
#         distances = np.array([mahalanobis_distance(x, centroid, cov_inv) for x in X for centroid in centroids])
#         distances = distances.reshape((len(X), k))
#         labels = np.argmin(distances, axis=1)
#         new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#     return labels, centroids

# def kmeans_cosine(X, k, max_iter=30):
#     centroids = X[np.random.choice(len(X), k, replace=False)]
#     for _ in range(max_iter):
#         distances = pairwise_distances(X, centroids, metric='cosine')
#         labels = np.argmin(distances, axis=1)
#         new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#     return labels, centroids

# def load_mnist_data():
#     mnist_train = MNIST(root='./data', train=True, download=True, transform=ToTensor())
#     mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    
#     X_train = mnist_train.data.numpy().reshape(-1, 28*28)
#     y_train = mnist_train.targets.numpy()
    
#     X_test = mnist_test.data.numpy().reshape(-1, 28*28)
#     y_test = mnist_test.targets.numpy()
    
#     return X_train, X_test, y_train, y_test

# def preprocess_data(X_train, X_test, n_components):
#     pca = PCA(n_components=n_components)
#     pca.fit(X_train)
#     X_train_pca = pca.transform(X_train)
#     X_test_pca = pca.transform(X_test)
#     return X_train_pca, X_test_pca

# def fit_gmm(X_train, y_train, K):
#     gmms = []
#     for digit in range(10):
#         X_digit = X_train[y_train == digit]
#         gmm = GaussianMixture(n_components=K, covariance_type='diag')
#         gmm.fit(X_digit)
#         gmms.append(gmm)
#     return gmms

# def predict_gmm(X_test, gmms, prior_probs):
#     n_samples = X_test.shape[0]
#     n_classes = len(gmms)
#     probs = np.zeros((n_samples, n_classes))
#     for i, gmm in enumerate(gmms):
#         probs[:, i] = gmm.score_samples(X_test) + np.log(prior_probs[i])
#     return np.argmax(probs, axis=1)

# def main():
#     X_train, X_test, y_train, y_test = load_mnist_data()
#     K_values = [5, 10, 20, 30]
    
#     for K in K_values:
#         # Apply PCA for dimensionality reduction
#         X_train_pca, X_test_pca = preprocess_data(X_train, X_test, n_components=5)
        
#         # Fit GMMs for each digit class
#         gmms = fit_gmm(X_train_pca, y_train, K)
        
#         # Calculate prior probabilities
#         prior_probs = np.array([np.mean(y_train == digit) for digit in range(10)])
        
#         # Predict using GMMs
#         y_pred = predict_gmm(X_test_pca, gmms, prior_probs)
        
#         # Calculate accuracy
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print(f"K={K}, Accuracy={accuracy:.4f}")

#     # Use k-means clustering with Mahalanobis distance
#     labels, centroids = kmeans_mahalanobis(X_train_pca, k=10)
#     print(labels, centroids)

#     # Use k-means clustering with Cosine distance
#     labels, centroids = kmeans_cosine(X_train_pca, k=10)
#     print(labels, centroids)

# if __name__ == "__main__":
#     main()







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
        if len(cluster_indices) > 0:
            cluster_class_counts = np.bincount(labels[cluster_indices], minlength=num_classes)
            most_common_class = np.argmax(cluster_class_counts)
            correct += cluster_class_counts[most_common_class]
    accuracy = (correct / len(labels)) * 100
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
        if np.allclose(centroids, new_centroids):  # Check for convergence
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
        if np.allclose(centroids, new_centroids):  # Check for convergence
            break
        centroids = new_centroids
    return labels

# Function to perform GMM clustering with cosine distance
def gmm_cosine(X, k, max_iter=100, tol=1e-4):
    # Initialize centroids using K-means++
    centroids = kmeans_plusplus_init(X, k)
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y_sampled))
    prior_probs = np.ones(k) / k
    
    # Initialize covariance matrices as identity matrices
    covariances = np.stack([np.eye(num_features) for _ in range(k)])
    
    # Initialize responsibilities matrix
    responsibilities = np.zeros((num_samples, k))
    
    for _ in range(max_iter):
        # E-step: Update responsibilities
        for j in range(k):
            diff = X - centroids[j]
            distances = np.sqrt(np.sum(np.square(diff), axis=1))
            responsibilities[:, j] = np.exp(-0.5 * distances)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        for j in range(k):
            weights = responsibilities[:, j]
            centroids[j] = np.average(X, axis=0, weights=weights)
            diff = X - centroids[j]
            covariances[j] = np.dot(weights * diff.T, diff) / weights.sum()
        
        # Compute log-likelihood
        log_likelihood = -np.sum(np.log(np.sum(responsibilities, axis=1)))
        
        # Check for convergence
        if _ > 0 and np.abs(log_likelihood - prev_log_likelihood) < tol * np.abs(log_likelihood):
            break
        prev_log_likelihood = log_likelihood
    
    return np.argmax(responsibilities, axis=1)

# Function to perform GMM clustering with Mahalanobis distance
def gmm_mahalanobis(X, k, max_iter=100, tol=1e-4):
    # Initialize centroids using K-means++
    centroids = kmeans_plusplus_init(X, k)
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y_sampled))
    prior_probs = np.ones(k) / k
    
    # Initialize covariance matrices as identity matrices
    covariances = np.stack([np.eye(num_features) for _ in range(k)])
    
    # Initialize responsibilities matrix
    responsibilities = np.zeros((num_samples, k))
    
    for _ in range(max_iter):
        # E-step: Update responsibilities
        for j in range(k):
            diff = X - centroids[j]
            distances = np.array([mahalanobis_distance(x, centroids[j], np.linalg.pinv(covariances[j])) for x in X])
            responsibilities[:, j] = np.exp(-0.5 * distances)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        for j in range(k):
            weights = responsibilities[:, j]
            centroids[j] = np.average(X, axis=0, weights=weights)
            diff = X - centroids[j]
            covariances[j] = np.dot(weights * diff.T, diff) / weights.sum()
        
        # Compute log-likelihood
        log_likelihood = -np.sum(np.log(np.sum(responsibilities, axis=1)))
        
        # Check for convergence
        if _ > 0 and np.abs(log_likelihood - prev_log_likelihood) < tol * np.abs(log_likelihood):
            break
        prev_log_likelihood = log_likelihood
    
    return np.argmax(responsibilities, axis=1)

# Function to calculate cluster consistency
def cluster_consistency(labels, cluster_labels, num_classes):
    cluster_consistency_values = []
    for cluster_label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        if len(cluster_indices) > 0:
            class_counts = np.bincount(labels[cluster_indices], minlength=num_classes)
            most_common_class_count = np.max(class_counts)
            total_cluster_points = len(cluster_indices)
            cluster_consistency = most_common_class_count / total_cluster_points
            cluster_consistency_values.append(cluster_consistency)
    overall_consistency = np.mean(cluster_consistency_values)
    return overall_consistency

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
k_values = [5, 10, 20, 40]

# Calculate clustering accuracy and consistency for each value of k using cosine distance
print("Results using cosine distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply GMM clustering with cosine distance
    labels_cosine = gmm_cosine(X_sampled, k)
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_cosine, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.4f}")
    
    # Calculate clustering consistency
    consistency = cluster_consistency(y_sampled, labels_cosine, num_classes=10)
    print(f"    Cluster consistency: {consistency:.4f}")

# Calculate clustering accuracy and consistency for each value of k using Mahalanobis distance
print("\nResults using Mahalanobis distance:")
for k in k_values:
    print(f"  k={k}:")
    
    # Apply GMM clustering with Mahalanobis distance
    labels_mahalanobis = gmm_mahalanobis(X_sampled, k)
    
    # Calculate clustering accuracy
    accuracy = clustering_accuracy(y_sampled, labels_mahalanobis, num_classes=10)
    print(f"    Clustering accuracy: {accuracy:.4f}")
    
    # Calculate clustering consistency
    consistency = cluster_consistency(y_sampled, labels_mahalanobis, num_classes=10)
    print(f"    Cluster consistency: {consistency:.4f}")
