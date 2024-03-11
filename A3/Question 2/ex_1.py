import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the VGG11 architecture
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(self.relu(self.bn4(self.conv4(x))))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(self.relu(self.bn6(self.conv6(x))))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.maxpool(self.relu(self.bn8(self.conv8(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load MNIST dataset and resize images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Modify trainset creation to cut to 1/10 of the original size
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainset_cut = torch.utils.data.Subset(trainset, range(6000))  # 1/10 of the original size
trainloader = torch.utils.data.DataLoader(trainset_cut, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG11().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Lists to store accuracy and loss values
train_loss_history = []
test_loss_history = []
train_accuracy_history = []
test_accuracy_history = []

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_loss_history.append(running_train_loss / len(trainloader))
    train_accuracy_history.append(correct_train / total_train)

    # Evaluate the model on test data
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_test_loss += loss.item()
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    test_loss_history.append(running_test_loss / len(testloader))
    test_accuracy_history.append(correct_test / total_test)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss_history[-1]:.4f}, "
          f"Train Acc: {train_accuracy_history[-1]*100:.2f}%, "
          f"Test Loss: {test_loss_history[-1]:.4f}, "
          f"Test Acc: {test_accuracy_history[-1]*100:.2f}%")

# Plotting the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(range(1, num_epochs + 1), test_accuracy_history, label='Test Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Number of Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_history, label='Training Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Number of Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, num_epochs + 1), test_loss_history, label='Test Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Test Loss vs Number of Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Number of Epochs')
plt.legend()

plt.tight_layout()
plt.show()
