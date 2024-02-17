# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from torchvision import datasets, transforms

# # Load MNIST training dataset
# mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# # Convert MNIST training dataset to numpy arrays
# X_train = mnist_train.data.numpy()
# y_train = mnist_train.targets.numpy()

# # Reshape the data to flatten the images
# X_train = X_train.reshape(X_train.shape[0], -1)

# # Filter dataset to contain only classes 3 and 4
# X_34 = X_train[(y_train == 3) | (y_train == 4)]
# y_34 = y_train[(y_train == 3) | (y_train == 4)]

# # Normalize data
# X_34 = X_34 / 255.0

# # Apply PCA to reduce dimensions to 2
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X_34)

# # Split data into train and test sets
# split = int(0.8 * len(X_2d))
# X_train, X_test = X_2d[:split], X_2d[split:]
# y_train, y_test = y_34[:split], y_34[split:]

# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2)**2))

# def predict(X_train, y_train, x_test, k):
#     distances = []
#     for i in range(len(X_train)):
#         dist = euclidean_distance(x_test, X_train[i])
#         distances.append((dist, y_train[i]))
#     distances = sorted(distances)[:k]
#     labels = [dist[1] for dist in distances]
#     return max(set(labels), key=labels.count)

# def k_nearest_neighbors(X_train, y_train, X_test, y_test, k):
#     correct = 0
#     for i in range(len(X_test)):
#         prediction = predict(X_train, y_train, X_test[i], k)
#         if prediction == y_test[i]:
#             correct += 1
#     accuracy = (correct / len(X_test)) * 100
#     return accuracy

# # Define values of k
# k_values = [1, 2, 3, 4, 5]

# # Compute accuracies for each value of k
# accuracies = []
# for k in k_values:
#     accuracy = k_nearest_neighbors(X_train, y_train, X_test, y_test, k)
#     accuracies.append(accuracy)

# # Plot accuracies for each value of k
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracies, marker='o')
# plt.title('Accuracy vs. Number of Neighbors (k)')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.xticks(k_values)
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
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

# Initialize lists to store k values and corresponding accuracy values
k_values = []
accuracy_values = []

# Evaluate k-nearest neighbor classifier for k = 1 to 5
for k in range(1, 6):
    correct_predictions = 0
    total_samples = len(X_test_34)
    for i, x_test in enumerate(X_test_34):
        prediction = knn_predict(X_train_34, y_train_34, x_test, k)
        if prediction == y_test_34[i]:
            correct_predictions += 1
    accuracy = (correct_predictions / total_samples) * 100
    k_values.append(k)
    accuracy_values.append(accuracy)

# Plot accuracy values vs k values
plt.plot(k_values, accuracy_values, marker='o')
plt.title('Accuracy vs k Values for k-Nearest Neighbor Classifier')
plt.xlabel('k Values')
plt.ylabel('Accuracy (%)')
plt.xticks(k_values)
plt.grid(True)
plt.show()