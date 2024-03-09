import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the logistic function (sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function for logistic regression
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Define the gradient of the cost function for logistic regression
def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * np.dot(X.T, (h - y))
    return grad

# Gradient descent algorithm for logistic regression
def gradient_descent(X, y, theta, alpha, num_epochs=100, tol=1e-5):
    m = len(y)
    costs = []
    for epoch in range(num_epochs):
        grad = gradient(X, y, theta)
        theta -= alpha * grad
        cost = cost_function(X, y, theta)
        costs.append(cost)
        # Check for convergence
        if epoch > 0 and abs(costs[epoch - 1] - costs[epoch]) < tol:
            break
    return theta, costs

# Load the MNIST dataset using torchvision
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define binary classification for digits/classes 3 and 4
classes = [3, 4]

# Filter dataset for selected classes
train_indices = torch.tensor([i for i in range(len(mnist_trainset.targets)) if mnist_trainset.targets[i] in classes])
test_indices = torch.tensor([i for i in range(len(mnist_testset.targets)) if mnist_testset.targets[i] in classes])

train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=len(train_indices), sampler=torch.utils.data.SubsetRandomSampler(train_indices))
test_loader = torch.utils.data.DataLoader(dataset=mnist_testset, batch_size=len(test_indices), sampler=torch.utils.data.SubsetRandomSampler(test_indices))

# Flatten the images and add bias term (with fixed non-zero value)
X_train = torch.flatten(next(iter(train_loader))[0], start_dim=1).numpy()
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_train[:, 0] = 1  # Fix the bias term to a non-zero value (e.g., 1)
y_train = (next(iter(train_loader))[1] == classes[0]).numpy().astype(int)

X_test = torch.flatten(next(iter(test_loader))[0], start_dim=1).numpy()
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_test[:, 0] = 1  # Fix the bias term to a non-zero value (e.g., 1)
y_test = (next(iter(test_loader))[1] == classes[0]).numpy().astype(int)

# Initialize parameters for logistic regression
theta_init = np.zeros(X_train.shape[1])

# Train logistic regression model using gradient descent
alpha = 0.1
num_epochs = 100
theta_trained, costs = gradient_descent(X_train, y_train, theta_init, alpha, num_epochs)

# Calculate predictions
def predict(X, theta):
    return sigmoid(np.dot(X, theta))

y_train_pred = predict(X_train, theta_trained)
y_test_pred = predict(X_test, theta_trained)

# Calculate errors and losses
train_error = np.mean((y_train_pred >= 0.5) != y_train)
test_error = np.mean((y_test_pred >= 0.5) != y_test)
train_loss = cost_function(X_train, y_train, theta_trained)
test_loss = cost_function(X_test, y_test, theta_trained)

# Print and plot the results
print("Training Error:", train_error)
print("Test Error:", test_error)
print("Training Loss:", train_loss)
print("Test Loss:", test_loss)

plt.figure(figsize=(12, 8))

# Plot training error
plt.subplot(221)
plt.plot(range(num_epochs), [(y_train_pred >= 0.5) != y_train for y_train_pred in predict(X_train, theta_trained)], label='Training Error')
plt.xlabel('Number of Epochs')
plt.ylabel('Error')
plt.title('Training Error vs Number of Epochs')

# Plot test error
plt.subplot(222)
plt.plot(range(num_epochs), [(y_test_pred >= 0.5) != y_test for y_test_pred in predict(X_test, theta_trained)], label='Test Error')
plt