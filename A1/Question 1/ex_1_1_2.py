import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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

# Calculate mean and covariance matrices
mean_class3 = np.mean(X_class3, axis=0)
covariance_class3 = np.cov(X_class3, rowvar=False)
covariance_class3_inv = np.linalg.inv(covariance_class3)

mean_class4 = np.mean(X_class4, axis=0)
covariance_class4 = np.cov(X_class4, rowvar=False)
covariance_class4_inv = np.linalg.inv(covariance_class4)

# MED classifier
def med_classifier(sample, mean1, mean2):
    d1 = np.linalg.norm(sample - mean1)
    d2 = np.linalg.norm(sample - mean2)
    if d1 < d2:
        return 3  # Class 3
    else:
        return 4  # Class 4

# Predict labels for training samples using MED classifier
med_predictions_train = [med_classifier(sample, mean_class3, mean_class4) for sample in X_train_pca]

# MMD classifier
def mahalanobis_distance(sample, mean, cov_inv):
    delta = sample - mean
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Predict labels for training samples using MMD classifier
mmd_predictions_train = []

for sample in X_train_pca:
    md_class3 = mahalanobis_distance(sample, mean_class3, covariance_class3_inv)
    md_class4 = mahalanobis_distance(sample, mean_class4, covariance_class4_inv)
    
    if md_class3 < md_class4:
        mmd_predictions_train.append(3)  # Class 3
    else:
        mmd_predictions_train.append(4)  # Class 4

# Calculate accuracy on the training set
med_accuracy_train = accuracy_score(y_train, med_predictions_train) * 100
mmd_accuracy_train = accuracy_score(y_train, mmd_predictions_train) * 100

print(f'MED Classifier Accuracy on Training Set: {med_accuracy_train:.2f}%')
print(f'MMD Classifier Accuracy on Training Set: {mmd_accuracy_train:.2f}%')

# Plotting decision boundaries on the training set
plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue')
plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red')
plt.scatter(mean_class3[0], mean_class3[1], marker='x', s=200, c='green', label='Class 3 Mean')
plt.scatter(mean_class4[0], mean_class4[1], marker='o', s=200, c='purple', label='Class 4 Mean')

plt.legend()
plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.show()

# Function to plot the decision boundary for MED classifier
def plot_med_decision_boundary(X, mean1, mean2):
    plt.scatter(X[:, 0], X[:, 1], c=med_predictions_train, cmap='RdYlBu', marker='o', edgecolor='k', s=30)
    plt.scatter(mean1[0], mean1[1], marker='x', s=200, c='green', label='Class 3 Mean')
    plt.scatter(mean2[0], mean2[1], marker='o', s=200, c='purple', label='Class 4 Mean')
    plt.legend()
    plt.title('Decision Boundary for MED Classifier')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Function to plot the decision boundary for MMD classifier
def plot_mmd_decision_boundary(X, mean1, mean2, cov_inv1, cov_inv2):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = np.array([mahalanobis_distance(np.array([x, y]), mean_class3, covariance_class3_inv) -
                  mahalanobis_distance(np.array([x, y]), mean_class4, covariance_class4_inv) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, levels=[0], colors='green', linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], c=mmd_predictions_train, cmap='RdYlBu', marker='o', edgecolor='k', s=30)
    plt.scatter(mean1[0], mean1[1], marker='x', s=200, c='green', label='Class 3 Mean')
    plt.scatter(mean2[0], mean2[1], marker='o', s=200, c='purple', label='Class 4 Mean')
    plt.legend()
    plt.title('Decision Boundary for MMD Classifier')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Plot decision boundary for MED classifier
plot_med_decision_boundary(X_train_pca, mean_class3, mean_class4)

# Plot decision boundary for MMD classifier
plot_mmd_decision_boundary(X_train_pca, mean_class3, mean_class4, covariance_class3_inv, covariance_class4_inv)
