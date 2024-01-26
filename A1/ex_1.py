import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Define a transform to normalize the pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download the MNIST training dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Extract features (flattened images) and labels
X_train = np.array([np.array(x).flatten() for x, _ in mnist_train])
y_train = np.array([y for _, y in mnist_train])

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Separate data for classes 3 and 4
X_class3 = X_train_pca[y_train == 3]
X_class4 = X_train_pca[y_train == 4]

# Implement MED and MMD classifiers
med_threshold = np.mean(X_train_pca, axis=0)
mmd_threshold = np.mean(X_class3, axis=0) + np.mean(X_class4, axis=0)

# Plotting
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
plt.title('Decision Boundaries for MED and MMD Classifiers')
plt.show()