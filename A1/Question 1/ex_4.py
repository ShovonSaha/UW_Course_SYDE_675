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
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Extract training data and labels
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
for images, labels in train_loader:
    images_train = images.numpy()
    labels_train = labels.numpy()

# Extract test data and labels
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
for images, labels in test_loader:
    images_test = images.numpy()
    labels_test = labels.numpy()

# Filter out only classes 3 and 4 for training
class3_images_train = images_train[labels_train == 3]
class4_images_train = images_train[labels_train == 4]

# Flatten the images
class3_flat_train = class3_images_train.reshape((len(class3_images_train), -1))
class4_flat_train = class4_images_train.reshape((len(class4_images_train), -1))

# Combine the flattened images for training
X_train = np.vstack((class3_flat_train, class4_flat_train))
y_train = np.hstack((np.zeros(len(class3_flat_train)), np.ones(len(class4_flat_train))))

# Perform PCA to reduce dimensionality to 20
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train)

# Separate data for classes 3 and 4 after PCA for training
X_class3_train = X_train_pca[y_train == 0]
X_class4_train = X_train_pca[y_train == 1]

# Calculate the mean of each class for training
mean_class3_train = np.mean(X_class3_train, axis=0)
mean_class4_train = np.mean(X_class4_train, axis=0)

# Calculate the covariance matrices for each class for training
cov_class3_train = np.cov(X_class3_train.T)
cov_class4_train = np.cov(X_class4_train.T)

# Calculate the inverse covariance matrices using eigenvalue decomposition for training
cov_eigvals_class3_train, cov_eigvecs_class3_train = np.linalg.eig(cov_class3_train)
cov_inv_class3_train = np.dot(cov_eigvecs_class3_train, np.dot(np.diag(1 / cov_eigvals_class3_train), cov_eigvecs_class3_train.T))

cov_eigvals_class4_train, cov_eigvecs_class4_train = np.linalg.eig(cov_class4_train)
cov_inv_class4_train = np.dot(cov_eigvecs_class4_train, np.dot(np.diag(1 / cov_eigvals_class4_train), cov_eigvecs_class4_train.T))

# Flatten the test images
class3_flat_test = images_test[labels_test == 3].reshape((len(images_test[labels_test == 3]), -1))
class4_flat_test = images_test[labels_test == 4].reshape((len(images_test[labels_test == 4]), -1))

# Combine the flattened images for testing
X_test = np.vstack((class3_flat_test, class4_flat_test))
y_test = np.hstack((np.zeros(len(class3_flat_test)), np.ones(len(class4_flat_test))))

# Perform PCA on test data
X_test_pca = pca.transform(X_test)

# Predict labels for test set using MED classifier
med_predictions_test = [med_classifier(sample, mean_class3_train, mean_class4_train) for sample in X_test_pca]

# Predict labels for test set using MMD classifier
mmd_predictions_test = [mmd_classifier(sample, mean_class3_train, cov_inv_class3_train, mean_class4_train, cov_inv_class4_train) for sample in X_test_pca]

# Calculate classification accuracy for MED classifier
med_accuracy = np.mean(med_predictions_test == y_test) * 100
print(f"MED Classifier Accuracy on Test Set: {med_accuracy:.2f}%")

# Calculate classification accuracy for MMD classifier
mmd_accuracy = np.mean(mmd_predictions_test == y_test) * 100
print(f"MMD Classifier Accuracy on Test Set: {mmd_accuracy:.2f}%")