import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearModel:
    def __init__(self, lr=0.001):
        torch.manual_seed(0)
        self.model = nn.Linear(1, 1, device=device)
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.weight = None
        self.bias = None

    def create_data(self, num=1000):
        global x_mean, y_mean, x_std, y_std
        x = torch.randn(num, 1) * 10
        y = 3 * x + 2 + torch.randn(num, 1)
        self.x_mean = x.mean()
        self.y_mean = y.mean()
        self.x_std = x.std()
        self.y_std = y.std()
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        return x.to(device), y.to(device)

    def train(self, epochs=5000):
        print("Training")
        x, y = self.create_data()

        for i in range(epochs):
            outputs = self.model(x)

            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss.item()}")

        self.weight = self.model.weight * self.y_std / self.x_std
        self.bias = (
            self.model.bias * self.y_std + self.y_mean - self.weight * self.x_mean
        )

        print("weight:", self.weight.item())
        print("bias:", self.bias.item())

    def load(self, path="linear.pth"):
        self.create_data()
        self.model.load_state_dict(torch.load(path))

    def save(self, path="linear.pth"):
        torch.save(self.model.state_dict(), path)

    def predict(self, input_x):
        x_tensor = torch.tensor([[input_x]], device=device)
        x_normalized = (x_tensor - self.x_mean) / self.x_std
        y_pred_norm = self.model(x_normalized)
        return (y_pred_norm * self.y_std + self.y_mean).item()


my_model = LinearModel(lr=0.001)

my_model.train()

print(my_model.predict(5))

my_model.save()
