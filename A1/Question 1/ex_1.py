# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from mnist import MNIST

# # Load the MNIST dataset
# mndata = MNIST('mnist_data')
# numbers, classes = mndata.load_training()

# # Filter out only classes 3 and 4
# class3_images = []
# class4_images = []

# for i, label in enumerate(classes):
#     if label == 3:
#         class3_images.append(numbers[i])
#     elif label == 4:
#         class4_images.append(numbers[i])

# # Convert lists to numpy arrays
# class3_images = np.array(class3_images)
# class4_images = np.array(class4_images)

# # Flatten the images
# class3_flat = class3_images.reshape((len(class3_images), -1))
# class4_flat = class4_images.reshape((len(class4_images), -1))

# # Combine the flattened images
# X = np.vstack((class3_flat, class4_flat))

# # Perform PCA to reduce dimensionality to 2
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot the classes
# plt.scatter(X_pca[:len(class3_flat), 0], X_pca[:len(class3_flat), 1], label='Class 3', c='blue', alpha=0.5)
# plt.scatter(X_pca[len(class3_flat):, 0], X_pca[len(class3_flat):, 1], label='Class 4', c='red', alpha=0.5)

# plt.title('MNIST Dataset - Classes 3 and 4')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend()
# plt.show()



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

# Implement MED classifier
med_threshold = np.mean(X_pca, axis=0)

# Implement MMD classifier
covariance_class3 = np.cov(X_class3, rowvar=False)
covariance_class3_inv = np.linalg.inv(covariance_class3)

covariance_class4 = np.cov(X_class4, rowvar=False)
covariance_class4_inv = np.linalg.inv(covariance_class4)

mmd_threshold = np.mean(X_class3, axis=0) + np.mean(X_class4, axis=0)

# Plotting decision boundaries on the training set
plt.scatter(X_class3[:, 0], X_class3[:, 1], label='Class 3', c='blue', alpha=0.5)
plt.scatter(X_class4[:, 0], X_class4[:, 1], label='Class 4', c='red', alpha=0.5)
plt.scatter(med_threshold[0], med_threshold[1], marker='x', s=200, c='green', label='MED Threshold', alpha=1.0)
plt.scatter(mmd_threshold[0], mmd_threshold[1], marker='o', s=200, c='purple', label='MMD Threshold', alpha=1.0)

# Decision boundary for MED classifier
x_med = np.linspace(np.min(X_pca[:, 0]), np.max(X_pca[:, 0]), 100)
y_med = (med_threshold[1] / med_threshold[0]) * x_med
plt.plot(x_med, y_med, label='MED Decision Boundary', linestyle='--', color='green')

# Decision boundary for MMD classifier
x_mmd = np.linspace(np.min(X_pca[:, 0]), np.max(X_pca[:, 0]), 100)
y_mmd = (mmd_threshold[1] / mmd_threshold[0]) * x_mmd
plt.plot(x_mmd, y_mmd, label='MMD Decision Boundary', linestyle='--', color='purple')

plt.title('Decision Boundaries for MED and MMD Classifiers - Training Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()