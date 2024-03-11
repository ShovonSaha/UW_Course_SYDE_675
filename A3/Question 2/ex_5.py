# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

# # Define the CNN architecture without dropout
# class CNNWithoutDropout(nn.Module):
#     def __init__(self):
#         super(CNNWithoutDropout, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
#         self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
#         self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
#         self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
#         self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.fc1 = nn.Linear(512, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 10)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = self.maxpool(self.relu(self.conv1(x)))
#         x = self.maxpool(self.relu(self.conv2(x)))
#         x = self.relu(self.conv3(x))
#         x = self.maxpool(self.relu(self.conv4(x)))
#         x = self.relu(self.conv5(x))
#         x = self.maxpool(self.relu(self.conv6(x)))
#         x = self.relu(self.conv7(x))
#         x = self.maxpool(self.relu(self.conv8(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Load the MNIST dataset
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# # Reduce the training dataset to 1/10
# train_dataset.data = train_dataset.data[:6000]
# train_dataset.targets = train_dataset.targets[:6000]

# # Define the data loaders
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# # Initialize the model, optimizer, and loss function
# model = CNNWithoutDropout()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# criterion = nn.CrossEntropyLoss()

# # Train the model
# num_epochs = 20
# train_acc_list = []
# test_acc_list = []

# for epoch in range(num_epochs):
#     model.train()
#     correct_train = 0
#     total_train = 0
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()
    
#     train_accuracy = 100 * correct_train / total_train
#     train_acc_list.append(train_accuracy)

#     model.eval()
#     correct_test = 0
#     total_test = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total_test += labels.size(0)
#             correct_test += (predicted == labels).sum().item()

#     test_accuracy = 100 * correct_test / total_test
#     test_acc_list.append(test_accuracy)

#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

# # Plot the training and test accuracies vs the number of epochs
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), train_acc_list, label='Training Accuracy')
# plt.plot(range(1, num_epochs + 1), test_acc_list, label='Test Accuracy')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy (%)')
# plt.title('Training and Test Accuracies vs Number of Epochs (SGD without Dropout)')
# plt.legend()
# plt.grid(True)
# plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the CNN architecture without dropout
class CNNWithoutDropout(nn.Module):
    def __init__(self):
        super(CNNWithoutDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.maxpool(self.relu(self.conv6(x)))
        x = self.relu(self.conv7(x))
        x = self.maxpool(self.relu(self.conv8(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Reduce the training dataset to 1/10
train_dataset.data = train_dataset.data[:6000]
train_dataset.targets = train_dataset.targets[:6000]

# Define the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model and loss function
model = CNNWithoutDropout()
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
num_epochs = 20
train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0
    total_train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        total_train_loss += loss.item()
    
    train_accuracy = 100 * correct_train / total_train
    train_acc_list.append(train_accuracy)
    train_loss_list.append(total_train_loss / len(train_loader))

    model.eval()
    correct_test = 0
    total_test = 0
    total_test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            total_test_loss += criterion(outputs, labels).item()

    test_accuracy = 100 * correct_test / total_test
    test_acc_list.append(test_accuracy)
    test_loss_list.append(total_test_loss / len(test_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

# Plot the training and test accuracies vs the number of epochs
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_acc_list, label='Test Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies vs Number of Epochs (Without Dropout)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_loss_list, label='Test Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Losses vs Number of Epochs (Without Dropout)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()