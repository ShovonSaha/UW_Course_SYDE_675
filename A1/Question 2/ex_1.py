import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Reshape the data to flatten the images
X_train = X_train.reshape(X_train.shape[0], -1)

# Filter dataset to contain only classes 3 and 4
X_34 = X_train[(y_train == 3) | (y_train == 4)]
y_34 = y_train[(y_train == 3) | (y_train == 4)]

# Normalize data
X_34 = X_34 / 255.0

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_34)

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
    correct = 0
    for i in range(len(X_test)):
        prediction = predict(X_train, y_train, X_test[i], k)
        predictions.append(prediction)
        if prediction == y_test[i]:
            correct += 1
    accuracy = (correct / len(X_test)) * 100
    return accuracy

# Define values of k
k_values = [1, 2, 3, 4, 5]

# Plot decision boundaries and accuracies for each value of k
plt.figure(figsize=(15, 10))
for i, k in enumerate(k_values, 1):
    # Compute accuracy
    accuracy = k_nearest_neighbors(X_train, y_train, X_test, k)

    # Train the model
    predictions = []
    for x in np.linspace(min(X_2d[:, 0]), max(X_2d[:, 0]), 100):
        for y in np.linspace(min(X_2d[:, 1]), max(X_2d[:, 1]), 100):
            pred = predict(X_train, y_train, np.array([x, y]), k)
            predictions.append((pred, x, y))
    predictions = np.array(predictions)

    # Plot decision boundaries
    plt.subplot(2, 3, i)
    plt.title(f'k = {k}, Accuracy: {accuracy:.2f}%')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_34, cmap='viridis')
    plt.scatter(predictions[:, 1], predictions[:, 2], c=predictions[:, 0], cmap='coolwarm', alpha=0.1)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()