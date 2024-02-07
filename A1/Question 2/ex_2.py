import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# # Load the MNIST dataset
# mnist = fetch_openml('mnist_784', version=1)
# X, y = mnist['data'], mnist['target']
# y = y.astype(np.uint8)  # Convert labels to integers

# # Filter dataset to contain only classes 3 and 4
# X_34 = X[(y == 3) | (y == 4)]
# y_34 = y[(y == 3) | (y == 4)]

# # Normalize data
# X_34 = X_34 / 255.0

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Extract features (flattened images) and labels
X_train = np.array([np.array(x).flatten() for x, _ in mnist_train])
y_train = np.array([y for _, y in mnist_train])

# Flatten and normalize the data
X_train_flat = X_train.reshape((X_train.shape[0], -1)) / 255.0

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train_flat)

# Split data into train and test sets
split = int(0.8 * len(X_2d))
X_train, X_test = X_2d[:split], X_2d[split:]
y_train, y_test = y_34[:split], y_34[split:]

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_test, X_train[i])
        distances.append((dist, y_train[i]))
    distances = sorted(distances)[:k]
    labels = [dist[1] for dist in distances]
    return max(set(labels), key=labels.count)

def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i], k))
    return predictions

# Define values of k
k_values = [1, 2, 3, 4, 5]

# Implement kNN classifier for each k value
for k in k_values:
    # Predict labels for test set
    y_pred = k_nearest_neighbors(X_train, y_train, X_test, k)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k}, Accuracy: {accuracy:.4f}")
