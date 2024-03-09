# import numpy as np
# import torch
# from sklearn.decomposition import PCA
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# # Define the logistic function (sigmoid)
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # Define the cost function for logistic regression
# def cost_function(X, y, theta):
#     m = len(y)
#     h = sigmoid(np.dot(X, theta))
#     cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
#     return cost

# # Define the gradient of the cost function for logistic regression
# def gradient(X, y, theta):
#     m = len(y)
#     h = sigmoid(np.dot(X, theta))
#     grad = (1 / m) * np.dot(X.T, (h - y))
#     return grad

# # Gradient descent algorithm for logistic regression
# def gradient_descent(X, y, theta, alpha, num_epochs=100, tol=1e-5):
#     m = len(y)
#     costs = []
#     for epoch in range(num_epochs):
#         grad = gradient(X, y, theta)
#         theta -= alpha * grad
#         cost = cost_function(X, y, theta)
#         costs.append(cost)
#         # Check for convergence
#         if epoch > 0 and abs(costs[epoch - 1] - costs[epoch]) < tol:
#             break
#     return theta, costs

# # Load the MNIST dataset using torchvision
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert images to tensors
#     transforms.Normalize((0.5,), (0.5,))  # Normalize images
# ])

# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# # Define binary classification for digits 3 and 4
# classes = [3, 4]

# # Filter dataset for selected classes
# train_indices = torch.tensor([i for i in range(len(mnist_trainset.targets)) if mnist_trainset.targets[i] in classes])
# test_indices = torch.tensor([i for i in range(len(mnist_testset.targets)) if mnist_testset.targets[i] in classes])

# train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=len(train_indices), sampler=torch.utils.data.SubsetRandomSampler(train_indices))
# test_loader = torch.utils.data.DataLoader(dataset=mnist_testset, batch_size=len(test_indices), sampler=torch.utils.data.SubsetRandomSampler(test_indices))

# # Flatten the images and add bias term
# X_train = torch.flatten(next(iter(train_loader))[0], start_dim=1).numpy()
# X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
# y_train = (next(iter(train_loader))[1] == classes[0]).numpy().astype(int)

# X_test = torch.flatten(next(iter(test_loader))[0], start_dim=1).numpy()
# X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
# y_test = (next(iter(test_loader))[1] == classes[0]).numpy().astype(int)

# # Perform PCA to reduce dimensionality to 2D
# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train[:, 1:])  # Exclude bias term from PCA
# X_train_pca = np.hstack((np.ones((X_train_pca.shape[0], 1)), X_train_pca))  # Add bias term back

# # Define the number of epochs
# num_epochs_list = [50, 100, 150]

# # Initialize lists to store values for plotting
# train_errors_list = []
# test_errors_list = []
# train_losses_list = []
# test_losses_list = []

# # Train logistic regression model using gradient descent for each number of epochs
# for num_epochs in num_epochs_list:
#     # Initialize parameters for logistic regression with correct shape
#     theta_init = np.zeros(X_train_pca.shape[1])

#     # Train logistic regression model using gradient descent
#     alpha = 0.1
#     theta_trained, costs = gradient_descent(X_train_pca, y_train, theta_init, alpha, num_epochs=num_epochs)

#     # Calculate predictions
#     def predict(X, theta):
#         return sigmoid(np.dot(X, theta))

#     y_train_pred = predict(X_train_pca, theta_trained)
#     y_test_pred = predict(X_test, theta_trained)

#     # Calculate errors and losses
#     train_error = np.mean((y_train_pred >= 0.5) != y_train)
#     test_error = np.mean((y_test_pred >= 0.5) != y_test)
#     train_loss = cost_function(X_train_pca, y_train, theta_trained)
#     test_loss = cost_function(X_test, y_test, theta_trained)

#     # Append values to lists for plotting
#     train_errors_list.append(train_error)
#     test_errors_list.append(test_error)
#     train_losses_list.append(train_loss)
#     test_losses_list.append(test_loss)

#     # Print the values
#     print(f"Number of Epochs: {num_epochs}")
#     print("Training Error:", train_error)
#     print("Test Error:", test_error)
#     print("Training Loss:", train_loss)
#     print("Test Loss:", test_loss)
#     print()

# # Create plots
# plt.figure(figsize=(12, 10))

# # Plot training error
# plt.subplot(221)
# plt.plot(num_epochs_list, train_errors_list, marker='o')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Training Error')
# plt.title('Training Error vs Number of Epochs')

# # Plot test error
# plt.subplot(222)
# plt.plot(num_epochs_list, test_errors_list, marker='o')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Test Error')
# plt.title('Test Error vs Number of Epochs')

# # Plot training loss
# plt.subplot(223)
# plt.plot(num_epochs_list, train_losses_list, marker='o')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Training Loss')
# plt.title('Training Loss vs Number of Epochs')

# # Plot test loss
# plt.subplot(224)
# plt.plot(num_epochs_list, test_losses_list, marker='o')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Test Loss')
# plt.title('Test Loss vs Number of Epochs')

# plt.tight_layout()
# plt.show()










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
num_epochs = 100

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

plt.tight_layout(h_pad=3)
plt.show()