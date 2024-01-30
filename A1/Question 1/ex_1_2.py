import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Load MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Extract features (flattened images) and labels
X_train = np.array([np.array(x).flatten() for x, _ in mnist_train])
y_train = np.array([y for _, y in mnist_train])

# Flatten and normalize the data
X_train_flat = X_train.reshape((X_train.shape[0], -1)) / 255.0

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flat)

# Separate data for classes 3 and 4
X_class3 = X_train_pca[y_train == 3]
X_class4 = X_train_pca[y_train == 4]

# Implement MED and MMD classifiers
med_threshold = np.mean(X_train_pca, axis=0)
mmd_threshold = np.mean(X_class3, axis=0) + np.mean(X_class4, axis=0)

# Calculate MED and MMD predictions
med_predictions = np.argmin(np.linalg.norm(X_train_pca - med_threshold, axis=1, keepdims=True), axis=1)
mmd_predictions = np.argmin(np.linalg.norm(X_train_pca - mmd_threshold, axis=1, keepdims=True), axis=1)

# Add columns for predicted labels
train_data = np.column_stack((X_train_pca, y_train, med_predictions, mmd_predictions))

# Plotting decision boundaries on the training set
plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue')
plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red')
plt.scatter(med_threshold[0], med_threshold[1], marker='x', s=200, c='green', label='MED Threshold')
plt.scatter(mmd_threshold[0], mmd_threshold[1], marker='o', s=200, c='purple', label='MMD Threshold')

# Decision boundary for MED classifier
x_med = np.linspace(np.min(X_train_pca[:, 0]), np.max(X_train_pca[:, 0]), 100)
y_med = (med_threshold[1] / med_threshold[0]) * x_med
plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='green')

# Decision boundary for MMD classifier
x_mmd = np.linspace(np.min(X_train_pca[:, 0]), np.max(X_train_pca[:, 0]), 100)
y_mmd = (mmd_threshold[1] / mmd_threshold[0]) * x_mmd
plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='purple')

plt.legend()
plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.show()

# Load MNIST test dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Extract features (flattened images) and labels
X_test = np.array([np.array(x).flatten() for x, _ in mnist_test])
y_test = np.array([y for _, y in mnist_test])

# Flatten and normalize the data
X_test_flat = X_test.reshape((X_test.shape[0], -1)) / 255.0

# Perform PCA on test data
X_test_pca = pca.transform(X_test_flat)

# Calculate MED and MMD predictions on the test set
med_predictions_test = np.argmin(np.linalg.norm(X_test_pca - med_threshold, axis=1, keepdims=True), axis=1)
mmd_predictions_test = np.argmin(np.linalg.norm(X_test_pca - mmd_threshold, axis=1, keepdims=True), axis=1)

# Add columns for predicted labels to the test set
test_data = np.column_stack((X_test_pca, y_test, med_predictions_test, mmd_predictions_test))

# Calculate accuracy on the test set
med_accuracy_test = (np.sum(test_data[:, -2] == test_data[:, -1]) / len(test_data)) * 100
mmd_accuracy_test = (np.sum(test_data[:, -3] == test_data[:, -1]) / len(test_data)) * 100

print(f'MED Classifier Accuracy on Test Set: {med_accuracy_test:.2f}%')
print(f'MMD Classifier Accuracy on Test Set: {mmd_accuracy_test:.2f}%')