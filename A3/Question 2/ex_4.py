import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the CNN architecture with Sigmoid activation
class CNN_Sigmoid(nn.Module):
    def __init__(self):
        super(CNN_Sigmoid, self).__init__()
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
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(self.sigmoid(self.conv1(x)))
        x = self.maxpool(self.sigmoid(self.conv2(x)))
        x = self.sigmoid(self.conv3(x))
        x = self.maxpool(self.sigmoid(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        x = self.maxpool(self.sigmoid(self.conv6(x)))
        x = self.sigmoid(self.conv7(x))
        x = self.maxpool(self.sigmoid(self.conv8(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.sigmoid(self.fc1(x)))
        x = self.dropout(self.sigmoid(self.fc2(x)))
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

# Initialize the model, optimizer, and loss function
model_sigmoid = CNN_Sigmoid()
optimizer_sigmoid = optim.Adam(model_sigmoid.parameters())
criterion = nn.CrossEntropyLoss()

# Train the model with Sigmoid activation
num_epochs = 20
train_acc_list_sigmoid = []
test_acc_list_sigmoid = []

for epoch in range(num_epochs):
    model_sigmoid.train()
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        optimizer_sigmoid.zero_grad()
        outputs = model_sigmoid(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_sigmoid.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    train_acc_list_sigmoid.append(train_accuracy)

    model_sigmoid.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_sigmoid(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_acc_list_sigmoid.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

# Plot the training and test accuracies vs the number of epochs for Sigmoid
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_acc_list_sigmoid, label='Training Accuracy (Sigmoid)')
plt.plot(range(1, num_epochs + 1), test_acc_list_sigmoid, label='Test Accuracy (Sigmoid)')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies vs Number of Epochs (Sigmoid)')
plt.legend()
plt.grid(True)
plt.show()