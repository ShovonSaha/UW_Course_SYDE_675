import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression with sigmoid activation
def cost_function_sigmoid(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Define the gradient descent algorithm for logistic regression with sigmoid activation
def gradient_descent_sigmoid(X, y, theta, alpha, num_epochs=100):
    m = len(y)
    costs = []
    for _ in range(num_epochs):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost = cost_function_sigmoid(X, y, theta)
        costs.append(cost)
    return theta, costs

# Define the cost function for logistic regression without activation
def cost_function_no_activation(X, y, theta):
    m = len(y)
    z = np.dot(X, theta)
    cost = (1 / m) * np.sum(np.square(z - y))
    return cost

# Define the gradient descent algorithm for logistic regression without activation
def gradient_descent_no_activation(X, y, theta, alpha, num_epochs=100):
    m = len(y)
    costs = []
    for _ in range(num_epochs):
        z = np.dot(X, theta)
        gradient = (1 / m) * np.dot(X.T, (z - y))
        theta -= alpha * gradient
        cost = cost_function_no_activation(X, y, theta)
        costs.append(cost)
    return theta, costs

# Load the MNIST dataset using torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

# Train logistic regression model using gradient descent with sigmoid activation
alpha = 0.01
num_epochs = 100
theta_trained_sigmoid, _ = gradient_descent_sigmoid(X_pca_bias, y_binary, theta_init, alpha, num_epochs)

# Train logistic regression model using gradient descent without activation
theta_trained_no_activation, _ = gradient_descent_no_activation(X_pca_bias, y_binary, theta_init, alpha, num_epochs)

# Calculate predictions for sigmoid activation
h_sigmoid = sigmoid(np.dot(X_pca_bias, theta_trained_sigmoid))
predictions_sigmoid = (h_sigmoid >= 0.5).astype(int)

# Calculate predictions for no activation
predictions_no_activation = (np.dot(X_pca_bias, theta_trained_no_activation) >= 0.5).astype(int)

# Evaluate performance
accuracy_sigmoid = np.mean(predictions_sigmoid == y_binary)
accuracy_no_activation = np.mean(predictions_no_activation == y_binary)

# Plot decision boundaries and accuracies
x_boundary = np.array([np.min(X_pca[:, 0]), np.max(X_pca[:, 0])])
y_boundary_sigmoid = -(theta_trained_sigmoid[0] + theta_trained_sigmoid[1] * x_boundary) / theta_trained_sigmoid[2]
y_boundary_no_activation = -(theta_trained_no_activation[0] + theta_trained_no_activation[1] * x_boundary) / theta_trained_no_activation[2]

plt.figure(figsize=(16, 6))

# Plot decision boundaries
plt.subplot(1, 2, 1)
plt.scatter(X_pca[y_binary == 0, 0], X_pca[y_binary == 0, 1], label='Digit 4', color='blue')
plt.scatter(X_pca[y_binary == 1, 0], X_pca[y_binary == 1, 1], label='Digit 3', color='red')
plt.plot(x_boundary, y_boundary_sigmoid, label='Sigmoid Activation', color='black', linestyle='--')
plt.plot(x_boundary, y_boundary_no_activation, label='No Activation', color='green', linestyle='-.')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression Decision Boundaries')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.bar(['Sigmoid Activation', 'No Activation'], [accuracy_sigmoid, accuracy_no_activation], color=['blue', 'green'])
plt.xlabel('Activation Function')
plt.ylabel('Accuracy')
plt.title('Model Accuracies')

plt.tight_layout()
plt.show()

print("Accuracy with sigmoid activation:", accuracy_sigmoid)
print("Accuracy without activation:", accuracy_no_activation)
