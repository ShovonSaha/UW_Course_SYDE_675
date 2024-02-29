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
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from torchvision import datasets, transforms

# Function to calculate Mahalanobis distance
def mahalanobis_distance(x, centroid, cov_inv):
    diff = x - centroid
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# K-means algorithm using Mahalanobis distance
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

# Function to initialize GMM parameters
def initialize_parameters(X, k):
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    kmeans_labels, kmeans_centers = kmeans_mahalanobis(X_scaled, k)
    return kmeans_labels, kmeans_centers, pca, scaler

# E-step: Compute responsibilities
def e_step(X, centers, cov_inv):
    n_samples = len(X)
    n_clusters = len(centers)
    responsibilities = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        for j in range(n_clusters):
            responsibilities[i, j] = np.exp(-0.5 * mahalanobis_distance(X[i], centers[j], cov_inv[j]))
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
    return responsibilities

# M-step: Update parameters
def m_step(X, responsibilities):
    n_clusters = responsibilities.shape[1]
    new_centers = np.zeros_like(centers)
    for j in range(n_clusters):
        new_centers[j] = np.dot(responsibilities[:, j], X) / np.sum(responsibilities[:, j])
    return new_centers

# Log-likelihood calculation
def log_likelihood(X, centers, cov_inv, responsibilities):
    n_samples = len(X)
    n_clusters = len(centers)
    log_likelihood = 0
    for i in range(n_samples):
        for j in range(n_clusters):
            log_likelihood += responsibilities[i, j] * mahalanobis_distance(X[i], centers[j], cov_inv[j])
    return log_likelihood

# Gaussian Mixture Model algorithm
def gmm(X, k, max_iter=100, tol=1e-5):
    # Initialization
    kmeans_labels, centers, pca, scaler = initialize_parameters(X, k)
    cov_inv = [np.linalg.pinv(np.cov(X[kmeans_labels == j].T)) for j in range(k)]
    n_samples = len(X)

    # EM iterations
    for _ in range(max_iter):
        # E-step
        responsibilities = e_step(X, centers, cov_inv)
        # M-step
        new_centers = m_step(X, responsibilities)
        # Log-likelihood calculation
        ll_old = log_likelihood(X, centers, cov_inv, responsibilities)
        # Update parameters
        centers = new_centers
        cov_inv = [np.linalg.pinv(np.cov(X[kmeans_labels == j].T)) for j in range(k)]
        # Check convergence
        ll_new = log_likelihood(X, centers, cov_inv, responsibilities)
        if np.abs(ll_new - ll_old) <= tol * np.abs(ll_new):
            break

    return centers

# Fetch MNIST dataset using torchvision
def fetch_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Main function to run GMM on MNIST dataset
def main():
    train_loader, test_loader = fetch_mnist()
    X_train = train_loader.dataset.data.numpy().reshape(-1, 784)
    X_test = test_loader.dataset.data.numpy().reshape(-1, 784)

    for k in [5, 10, 20, 30]:
        # Fit GMM
        centers = gmm(X_train, k)
        # Perform classification using Bayes classifier
        # Report accuracy on test set

if __name__ == "__main__":
    main()
