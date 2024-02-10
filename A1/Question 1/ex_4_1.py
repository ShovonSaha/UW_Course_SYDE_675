import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch

# Define MED and MMD classifiers
def med_classifier(sample, mean1, mean2):
    d1 = np.linalg.norm(sample - mean1)
    d2 = np.linalg.norm(sample - mean2)
    if d1 < d2:
        return 0
    else:
        return 1

def mmd_classifier(sample, mean1, cov_inv1, mean2, cov_inv2):
    md_class1 = np.sqrt(np.dot(np.dot((sample - mean1).T, cov_inv1), sample - mean1))
    md_class2 = np.sqrt(np.dot(np.dot((sample - mean2).T, cov_inv2), sample - mean2))
    if md_class1 < md_class2:
        return 0
    else:
        return 1

# Load the MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Extract training data and labels
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
for images, labels in train_loader:
    images_train = images.numpy()
    labels_train = labels.numpy()

# Filter out only classes 3 and 4 for training
class3_images_train = images_train[labels_train == 3]
class4_images_train = images_train[labels_train == 4]

# Flatten the images
class3_flat_train = class3_images_train.reshape((len(class3_images_train), -1))
class4_flat_train = class4_images_train.reshape((len(class4_images_train), -1))

# Combine the flattened images for training
X_train = np.vstack((class3_flat_train, class4_flat_train))
y_train = np.hstack((np.zeros(len(class3_flat_train)), np.ones(len(class4_flat_train))))

# Perform PCA to reduce dimensionality to 20 for plotting
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)

# Separate data for classes 3 and 4 after PCA for training
X_class3_train = X_train_pca[y_train == 0]
X_class4_train = X_train_pca[y_train == 1]

# Calculate the mean of each class for training
mean_class3_train = np.mean(X_class3_train, axis=0)
mean_class4_train = np.mean(X_class4_train, axis=0)

# Calculate the slope of the MED decision boundary
slope_med = -(mean_class4_train[0] - mean_class3_train[0]) / (mean_class4_train[1] - mean_class3_train[1])

# Calculate the midpoint of the MED decision boundary
midpoint_med = [(mean_class3_train[0] + mean_class4_train[0]) / 2, (mean_class3_train[1] + mean_class4_train[1]) / 2]

# Calculate the covariance matrices for each class for training
cov_class3_train = np.cov(X_class3_train.T)
cov_class4_train = np.cov(X_class4_train.T)

# Calculate the inverse covariance matrices using eigenvalue decomposition for training
cov_eigvals_class3_train, cov_eigvecs_class3_train = np.linalg.eig(cov_class3_train)
cov_inv_class3_train = np.dot(cov_eigvecs_class3_train, np.dot(np.diag(1 / cov_eigvals_class3_train), cov_eigvecs_class3_train.T))

cov_eigvals_class4_train, cov_eigvecs_class4_train = np.linalg.eig(cov_class4_train)
cov_inv_class4_train = np.dot(cov_eigvecs_class4_train, np.dot(np.diag(1 / cov_eigvals_class4_train), cov_eigvecs_class4_train.T))

# Calculate the mean difference between the means of the two classes
mean_diff = mean_class3_train - mean_class4_train

# Calculate the average inverse covariance matrix
cov_avg_inv = (cov_inv_class3_train + cov_inv_class4_train) / 2

# Calculate the slope of the MMD decision boundary
slope_mmd = -cov_avg_inv[0, 0] / cov_avg_inv[0, 1]

# Calculate the intercept of the MMD decision boundary
intercept_mmd = np.dot(cov_avg_inv, mean_diff)[1]

# Plotting decision boundaries and training data
plt.scatter(X_class3_train[:, 0], X_class3_train[:, 1], label='Class 3', c='blue', alpha=0.5)
plt.scatter(X_class4_train[:, 0], X_class4_train[:, 1], label='Class 4', c='red', alpha=0.5)
plt.scatter(mean_class3_train[0], mean_class3_train[1], marker='x', s=200, c='green', label='Class 3 Mean')
plt.scatter(mean_class4_train[0], mean_class4_train[1], marker='o', s=200, c='purple', label='Class 4 Mean')

# Plot the MED decision boundary
x_med = np.linspace(np.min(X_train_pca[:, 0]), np.max(X_train_pca[:, 0]), 100)
y_med = slope_med * (x_med - midpoint_med[0]) + midpoint_med[1]
plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='black')

# Plot the MMD decision boundary
x_mmd = np.linspace(np.min(X_train_pca[:, 0]), np.max(X_train_pca[:, 0]), 100)
y_mmd = slope_mmd * x_mmd + intercept_mmd
plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='green')

plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()