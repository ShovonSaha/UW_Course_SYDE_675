import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Define a transform to flatten the images
flatten_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

# Download the MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=flatten_transform)

# Extract features (flattened images) and labels
X_train = np.array([np.array(x) for x, _ in mnist_train])
y_train = np.array([y for _, y in mnist_train])

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Separate data for each class
X_class0 = X_train_pca[y_train == 3]
X_class1 = X_train_pca[y_train == 4]
# Continue this for other classes if needed

# Plotting
plt.scatter(X_class0[:, 0], X_class0[:, 1], label='Class 0', c='blue')
plt.scatter(X_class1[:, 0], X_class1[:, 1], label='Class 1', c='red')
# Continue adding scatter plots for other classes if needed
plt.legend()
plt.title('PCA of MNIST Dataset')
plt.show()