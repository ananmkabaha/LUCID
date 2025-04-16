import torch.nn as nn
import torch.nn.functional as F

class FC_2x10(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class FC_2x50(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FC_2x100(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FC_4x30(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(self.dim, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, 2)
        self.m = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.reshape(-1, self.dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class CNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(1, 3), stride=1, padding='valid')
        self.conv2 = nn.Conv2d(6, 6, kernel_size=(1, 3), stride=1, padding='valid')
        self.flatten1 = nn.Flatten()
        if dim == 23:
            self.fc1 = nn.Linear(114, 2)
        elif dim == 21:
            self.fc1 = nn.Linear(102, 2)
        elif dim == 15:
            self.fc1 = nn.Linear(66, 2)
        elif dim == 7:
            self.fc1 = nn.Linear(18, 2)
        else:
            assert False, "Not suppoted diminsion, please extend the code."
        self.m = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = x.view(-1, 1, 1, self.dim)
        x = F.relu(self.conv1(x))
        x = self.m(x)
        x = F.relu(self.conv2(x))
        x = self.m(x)
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = self.fc2(x)
        return x
