import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(28*28,64)
        self.f2 = nn.Linear(64,64)
        self.f3 = nn.Linear(64,10)

    def forward(self , x):
        x = x.view(-1, 28*28)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x 
