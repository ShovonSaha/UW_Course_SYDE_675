import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Define the logistic function (sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the gradient of the logistic function
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define the cost function for logistic regression
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Define the gradient descent algorithm
def gradient_descent(X, y, theta, alpha, num_epochs=100):
    m = len(y)
    costs = []
    for _ in range(num_epochs):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        costs.append(cost)
    return theta, costs

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
X, y = mnist_trainset.data.numpy(), mnist_trainset.targets.numpy()

# Keep only the data for digits 3 and 4
X_filtered = X[(y == 3) | (y == 4)]
y_filtered = y[(y == 3) | (y == 4)]

# Flatten the images and standardize the features
X_flat = X_filtered.reshape(X_filtered.shape[0], -1)
X_flat = X_flat / 255.0

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat)

# Add bias term to the feature matrix
X_pca_bias = np.hstack((np.ones((X_pca.shape[0], 1)), X_pca))

# Define labels for binary classification (3 vs 4)
y_binary = (y_filtered == 3).astype(int)

# Initialize parameters for logistic regression
theta_init = np.zeros(X_pca_bias.shape[1])

# Train logistic regression model using gradient descent
alpha = 0.01
# num_epochs_list = [50, 100, 150]
num_epochs_list = [100]

for num_epochs in num_epochs_list:
    theta_trained, costs = gradient_descent(X_pca_bias, y_binary, theta_init, alpha, num_epochs=num_epochs)

    # Calculate predictions
    y_pred = sigmoid(np.dot(X_pca_bias, theta_trained))
    y_pred_class = (y_pred >= 0.5).astype(int)

    # Calculate errors
    train_error = np.mean(y_pred_class != y_binary)
    train_loss = cost_function(X_pca_bias, y_binary, theta_trained)

    print("Number of Epochs:", num_epochs)
    print("Training Error:", train_error)
    print("Training Loss:", train_loss)

    # Plot the results
    plt.figure(figsize=(12, 12))

    # Plot training error
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), [np.mean(y_pred_class[:i] != y_binary[:i]) for i in range(1, num_epochs + 1)])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Error')
    plt.title('Training Error vs Number of Epochs')

    # Plot test error
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), [np.mean((sigmoid(np.dot(X_pca_bias[:i], theta_trained)) >= 0.5).astype(int) != y_binary[:i]) for i in range(1, num_epochs + 1)])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Test Error')
    plt.title('Test Error vs Number of Epochs')

    # Plot training loss
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs + 1), costs)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Number of Epochs')

    # Calculate test loss
    test_loss = [cost_function(X_pca_bias[:i], y_binary[:i], theta_trained) for i in range(1, num_epochs + 1)]

    # Plot test loss
    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), test_loss)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Number of Epochs')

    plt.tight_layout()
    plt.show()
