import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from scipy.stats import multivariate_normal

# Load MNIST training and test datasets
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Convert MNIST training dataset to numpy arrays
X_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

# Convert MNIST test dataset to numpy arrays
X_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Reshape the data to flatten the images to 784x1 vectors
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Use PCA to reduce dimensionality to 2x1 vectors
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Filter training and test datasets to contain only classes 3 and 4
X_train_34 = X_train_pca[(y_train == 3) | (y_train == 4)]
y_train_34 = y_train[(y_train == 3) | (y_train == 4)]
X_test_34 = X_test_pca[(y_test == 3) | (y_test == 4)]
y_test_34 = y_test[(y_test == 3) | (y_test == 4)]

# Calculate sample mean and covariance matrix for each class
mean_class3 = np.mean(X_train_34[y_train_34 == 3], axis=0)
cov_class3 = np.cov(X_train_34[y_train_34 == 3].T)
mean_class4 = np.mean(X_train_34[y_train_34 == 4], axis=0)
cov_class4 = np.cov(X_train_34[y_train_34 == 4].T)

# Define PDFs for each class
pdf_class3 = multivariate_normal(mean=mean_class3, cov=cov_class3)
pdf_class4 = multivariate_normal(mean=mean_class4, cov=cov_class4)

# Implement Maximum Likelihood (ML) classifier
def ml_classifier(sample):
    if pdf_class3.pdf(sample) > pdf_class4.pdf(sample):
        return 3
    else:
        return 4

# Evaluate ML classifier on test set
correct_predictions = 0
total_samples = len(X_test_34)

for sample, true_label in zip(X_test_34, y_test_34):
    prediction = ml_classifier(sample)
    if prediction == true_label:
        correct_predictions += 1

accuracy_test = (correct_predictions / total_samples) * 100
print("ML Classifier Accuracy on Test Set: {:.2f}%".format(accuracy_test))

# Prior probabilities
p_class3 = 0.58
p_class4 = 0.42

# Implement Maximum A Posteriori (MAP) classifier
def map_classifier(sample):
    likelihood_class3 = pdf_class3.pdf(sample)
    likelihood_class4 = pdf_class4.pdf(sample)
    
    # Calculate posterior probabilities
    posterior_class3 = (likelihood_class3 * p_class3) / ((likelihood_class3 * p_class3) + (likelihood_class4 * p_class4))
    posterior_class4 = (likelihood_class4 * p_class4) / ((likelihood_class3 * p_class3) + (likelihood_class4 * p_class4))
    
    if posterior_class3 > posterior_class4:
        return 3
    else:
        return 4

# Evaluate MAP classifier on test set
correct_predictions = 0

for sample, true_label in zip(X_test_34, y_test_34):
    prediction = map_classifier(sample)
    if prediction == true_label:
        correct_predictions += 1

accuracy_test_map = (correct_predictions / total_samples) * 100
print("MAP Classifier Accuracy on Test Set: {:.2f}%".format(accuracy_test_map))