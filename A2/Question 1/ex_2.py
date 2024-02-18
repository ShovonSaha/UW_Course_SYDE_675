import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torchvision import datasets, transforms

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Reshape the data to flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)

# Define the list of k values to try
k_values = [5, 10, 20, 40, 200]

# Iterate over each k value
for k in k_values:
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)

    # Fit the model to the data
    kmeans.fit(X_train_flattened)

    # Evaluate clustering results
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_train_flattened, kmeans.labels_, random_state=42)

    print(f"K={k}:")
    print(f"Inertia: {inertia}")
    print(f"Silhouette Score: {silhouette}\n")

    # Optionally, you can visualize clustering results or centroids here
