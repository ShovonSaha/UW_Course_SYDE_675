import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mnist import MNIST

# Load the MNIST dataset
mndata = MNIST('mnist_data')
numbers, classes = mndata.load_training()

# Filter out only classes 3 and 4
class3_images = []
class4_images = []

for i, label in enumerate(classes):
    if label == 3:
        class3_images.append(numbers[i])
    elif label == 4:
        class4_images.append(numbers[i])

# Convert lists to numpy arrays
class3_images = np.array(class3_images)
class4_images = np.array(class4_images)

# Flatten the images
class3_flat = class3_images.reshape((len(class3_images), -1))
class4_flat = class4_images.reshape((len(class4_images), -1))

# Combine the flattened images
X = np.vstack((class3_flat, class4_flat))
y = np.hstack((np.zeros(len(class3_flat)), np.ones(len(class4_flat))))

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Separate data for classes 3 and 4 after PCA
X_class3 = X_pca[y == 0]
X_class4 = X_pca[y == 1]

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

# Calculate the mean of each class
mean_class3 = np.mean(X_class3, axis=0)
mean_class4 = np.mean(X_class4, axis=0)

# Calculate the covariance matrices for each class
cov_class3 = np.cov(X_class3.T)
cov_class4 = np.cov(X_class4.T)

# Calculate the inverse covariance matrices
cov_inv_class3 = np.linalg.inv(cov_class3)
cov_inv_class4 = np.linalg.inv(cov_class4)

# Calculate the slope of the MED decision boundary
slope_med = -(mean_class4[0] - mean_class3[0]) / (mean_class4[1] - mean_class3[1])

# Calculate the midpoint of the MED decision boundary
midpoint_med = [(mean_class3[0] + mean_class4[0]) / 2, (mean_class3[1] + mean_class4[1]) / 2]

# Calculate the MMD decision boundary
mean_diff = mean_class3 - mean_class4
cov_avg_inv = (cov_inv_class3 + cov_inv_class4) / 2

# Calculate the slope of the MMD decision boundary
slope_mmd = -cov_avg_inv[0, 0] / cov_avg_inv[0, 1]

# Calculate the intercept of the MMD decision boundary
intercept_mmd = np.dot(cov_avg_inv, mean_diff)[1]

# Plotting decision boundaries and training data
plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue', alpha=0.5)
plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red', alpha=0.5)
plt.scatter(mean_class3[0], mean_class3[1], marker='x', s=200, c='green', label='Class 3 Mean')
plt.scatter(mean_class4[0], mean_class4[1], marker='o', s=200, c='purple', label='Class 4 Mean')

# Plot the MED decision boundary
x_med = np.linspace(np.min(X_pca[:, 0]), np.max(X_pca[:, 0]), 100)
y_med = slope_med * (x_med - midpoint_med[0]) + midpoint_med[1]
plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='black')

# Plot the MMD decision boundary
x_mmd = np.linspace(np.min(X_pca[:, 0]), np.max(X_pca[:, 0]), 100)
y_mmd = slope_mmd * x_mmd + intercept_mmd
plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='green')

plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()