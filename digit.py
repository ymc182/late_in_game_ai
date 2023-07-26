import torch
import torch.functional as F
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


trainset = datasets.MNIST("./DATASET/MNIST_DATA/", transform=transform)


# Download and load the test data
testset = datasets.MNIST("~/.pytorch/MNIST_data/", transform=transform)

testloader = DataLoader(testset, batch_size=64, shuffle=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 28x28 pixel = 28-5+1 = 24x24
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        # after pool1 = 24/2 = 12x12
        self.conv2 = nn.Conv2d(32, 64, 5)
        # 12-5+1 /2 = 4x4
        self.pool2 = nn.MaxPool2d(2)
        # 64 output = 64x4x4 = 1024
        self.fc1 = nn.Linear(1024, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # flatten the tensor
        x = self.fc1(x)
        return x

    def train_model(self, trainloader, criterion, optimizer, epochs):
        model = self
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get inputs
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero parms gradients
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if i % 500 == 0:
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0

        print("Finished Training")


model = DigitModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model.train_model(trainloader, criterion, optimizer, 10)

torch.save(model.state_dict(), "digit.pth")
