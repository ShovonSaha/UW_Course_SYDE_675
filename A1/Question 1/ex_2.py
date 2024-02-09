import numpy as np
from mnist import MNIST
from sklearn.decomposition import PCA

# Load the MNIST test dataset
mndata = MNIST('mnist_data')
numbers, classes = mndata.load_testing()

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
X_test = np.vstack((class3_flat, class4_flat))
y_test = np.hstack((np.zeros(len(class3_flat)), np.ones(len(class4_flat))))

# Perform PCA on test data
X_test_pca = pca.transform(X_test)

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

# Calculate mean and covariance matrices for training set
mean_class3_train = np.mean(X_class3, axis=0)
mean_class4_train = np.mean(X_class4, axis=0)
cov_class3_train = np.cov(X_class3.T)
cov_class4_train = np.cov(X_class4.T)
cov_inv_class3_train = np.linalg.inv(cov_class3_train)
cov_inv_class4_train = np.linalg.inv(cov_class4_train)

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