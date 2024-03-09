import numpy as np
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
X_34_train = X_train[(y_train == 3) | (y_train == 4)]
y_34_train = y_train[(y_train == 3) | (y_train == 4)]

# Normalize data
X_34_train = X_34_train / 255.0

# Apply PCA to reduce dimensions to 1
pca = PCA(n_components=1)
X_1d_train = pca.fit_transform(X_34_train)

# Load MNIST test dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Convert MNIST test dataset to numpy arrays
X_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Reshape the data to flatten the images
X_test = X_test.reshape(X_test.shape[0], -1)

# Filter dataset to contain only classes 3 and 4
X_34_test = X_test[(y_test == 3) | (y_test == 4)]
y_34_test = y_test[(y_test == 3) | (y_test == 4)]

# Normalize test data
X_34_test = X_34_test / 255.0

# Apply PCA to reduce dimensions to 1 for test data
X_1d_test = pca.transform(X_34_test)

# Define the Gaussian kernel density estimation function
def gaussian_kernel(x, x_i, bandwidth):
    return np.exp(-0.5 * ((x - x_i) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

def kernel_density_estimation(data, test_data, bandwidth):
    n = len(data)
    kde_values = np.zeros(len(test_data))
    for i, x in enumerate(test_data):
        kde_values[i] = np.sum(gaussian_kernel(x, data, bandwidth)) / n
    return kde_values

# Perform kernel density estimation for class 3
kde_values_class3 = kernel_density_estimation(X_1d_train[y_34_train == 3], X_1d_test, bandwidth=20)

# Perform kernel density estimation for class 4
kde_values_class4 = kernel_density_estimation(X_1d_train[y_34_train == 4], X_1d_test, bandwidth=20)

# Define the ML-based classifier using probability estimation
def ml_classifier(kde_values_class3, kde_values_class4):
    prob_class3 = kde_values_class3
    prob_class4 = kde_values_class4
    return np.argmax(np.vstack((prob_class3, prob_class4)), axis=0) + 3

# Classify test set samples using ML-based classifier
ml_predictions = ml_classifier(kde_values_class3, kde_values_class4)

# Calculate accuracy
accuracy_kernel_density_ml = np.sum(ml_predictions == y_34_test) / len(y_34_test) * 100

# Print the accuracy
print("Accuracy of ML Classifier using Kernel Density Estimation: {:.2f}%".format(accuracy_kernel_density_ml))