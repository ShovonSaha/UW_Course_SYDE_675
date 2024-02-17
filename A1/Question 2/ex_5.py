import os
import numpy as np

# Define the directory where the data files are located
data_dir = r'C:\Users\Nazia\OneDrive\Documents\Shovon\UW_Course_SYDE_675\A1'

# Load the training and test data
X_train = np.genfromtxt(os.path.join(data_dir, 'X_train_F.csv'), delimiter=',')
X_test = np.genfromtxt(os.path.join(data_dir, 'X_test_F.csv'), delimiter=',')
Y_train = np.genfromtxt(os.path.join(data_dir, 'Y_train_F.csv'), delimiter=',')
Y_test = np.genfromtxt(os.path.join(data_dir, 'Y_test_F.csv'), delimiter=',')

# Function to calculate Euclidean distance between two points
def custom_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Function to perform k-nearest neighbor regression
def knn_regression(X_train, Y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [custom_distance(test_point, train_point) for train_point in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = Y_train[nearest_indices]
        prediction = np.mean(nearest_labels)
        predictions.append(prediction)
    return predictions

# Evaluate k-nearest neighbor regression for k = 1 to 3
for k in range(1, 4):
    predictions = knn_regression(X_train, Y_train, X_test, k)
    mse = np.mean((predictions - Y_test) ** 2)
    print("k = {}, Mean Squared Error: {:.4f}".format(k, mse))