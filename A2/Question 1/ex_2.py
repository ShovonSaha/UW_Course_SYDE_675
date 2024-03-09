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

# Separate data for classes 3 and 4 after PCA
X_class3_train = X_1d_train[y_34_train == 3]
X_class4_train = X_1d_train[y_34_train == 4]

# Calculate the mean and variance of each class
mean_class3_train = np.mean(X_class3_train)
mean_class4_train = np.mean(X_class4_train)
var_class3_train = np.var(X_class3_train)
var_class4_train = np.var(X_class4_train)

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

# Define the standard Gaussian ML classifier
def standard_gaussian_ml_classifier(data, mean3, var3, mean4, var4):
    p3gauss = (1 / (np.sqrt(2 * np.pi * var3))) * np.exp(-0.5 * ((data - mean3) ** 2) / var3)
    p4gauss = (1 / (np.sqrt(2 * np.pi * var4))) * np.exp(-0.5 * ((data - mean4) ** 2) / var4)
    return p3gauss, p4gauss

# Define the unique exponential-gauss ML classifier
def unique_exponential_gauss_ml_classifier(data, mean3, mean4):
    lambda3 = 1 / mean3
    lambda4 = 1 / mean4
    exponential3 = lambda3 * np.exp(-lambda3 * data)
    exponential4 = lambda4 * np.exp(-lambda4 * data)
    uniqueprob3 = 0.5 * p3gauss * exponential3
    uniqueprob4 = 0.5 * p4gauss * exponential4
    return uniqueprob3, uniqueprob4

# Perform classification using both classifiers on test set
experimentallabelML = np.zeros(len(X_1d_test))
expergausslabelML = np.zeros(len(X_1d_test))
for h in range(len(X_1d_test)):
    # Standard Gaussian ML classifier
    p3gauss, p4gauss = standard_gaussian_ml_classifier(X_1d_test[h], mean_class3_train, var_class3_train, mean_class4_train, var_class4_train)

    # Unique exponential-gauss ML classifier
    uniqueprob3, uniqueprob4 = unique_exponential_gauss_ml_classifier(X_1d_test[h], mean_class3_train, mean_class4_train)

    # Choose the class with higher probability for each classifier
    if p3gauss > p4gauss:
        experimentallabelML[h] = 3
    else:
        experimentallabelML[h] = 4

    if uniqueprob3 > uniqueprob4:
        expergausslabelML[h] = 3
    else:
        expergausslabelML[h] = 4

# Calculate accuracy for both classifiers
accuracy_standard_gaussian_ml = np.sum(experimentallabelML == y_34_test) / len(y_34_test)
accuracy_unique_exponential_gauss_ml = np.sum(expergausslabelML == y_34_test) / len(y_34_test)

# Print the accuracies
print("Accuracy of Standard Gaussian ML Classifier:", accuracy_standard_gaussian_ml)
print("Accuracy of Unique Exponential-Gauss ML Classifier:", accuracy_unique_exponential_gauss_ml)

# Compare and explain which classifier is the best
if accuracy_standard_gaussian_ml > accuracy_unique_exponential_gauss_ml:
    print("The Standard Gaussian ML Classifier is the best.")
    print("This might be because the assumption of a simple Gaussian distribution fits the data better.")
else:
    print("The Unique Exponential-Gauss ML Classifier is the best.")
    print("This might be because the combined Gaussian and exponential distribution better captures the underlying data distribution.")
