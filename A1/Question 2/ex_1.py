import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# Function to calculate Euclidean distance between two points
def custom_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to predict label using k-nearest neighbors
def knn_predict(X_train, y_train, x_test, k):
    distances = np.linalg.norm(X_train - x_test, axis=1)  # Calculate distances to all training points
    nearest_indices = np.argsort(distances)[:k]  # Get indices of k nearest neighbors
    nearest_labels = y_train[nearest_indices]  # Get labels of k nearest neighbors
    unique_labels, counts = np.unique(nearest_labels, return_counts=True)  # Count occurrences of each label
    prediction = unique_labels[np.argmax(counts)]  # Predict the label with maximum occurrences
    return prediction

# Load MNIST training and test datasets
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Convert MNIST test dataset to numpy arrays
X_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Reshape the data to flatten the images to 784x1 vectors
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Use PCA to reduce dimensionality to 2x1 vectors
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Filter training and test datasets to contain only classes 3 and 4
X_train_34 = X_train_pca[(y_train == 3) | (y_train == 4)]
y_train_34 = y_train[(y_train == 3) | (y_train == 4)]
X_test_34 = X_test_pca[(y_test == 3) | (y_test == 4)]
y_test_34 = y_test[(y_test == 3) | (y_test == 4)]

# Evaluate k-nearest neighbor classifier for k = 1 to 5
for k in range(1, 6):
    correct_predictions = 0
    total_samples = len(X_test_34)
    for i, x_test in enumerate(X_test_34):
        prediction = knn_predict(X_train_34, y_train_34, x_test, k)
        if prediction == y_test_34[i]:
            correct_predictions += 1
    accuracy = (correct_predictions / total_samples) * 100
    print("k = {}, Accuracy: {:.2f}%".format(k, accuracy))