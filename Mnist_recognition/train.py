import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Net

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("./data", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

train_loader = get_data_loader(True)


Epoch = []
Loss = []

min_loss = float('inf')
epoch_range = 5 

for epoch in range(epoch_range):
    running_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    avg_loss = running_loss / len(train_loader)

    if avg_loss < min_loss:
        torch.save(net.state_dict(), 'parameters/model.pth')
        min_loss = avg_loss

    Loss.append(avg_loss)
    Epoch.append(epoch+1)
    print(f"Epoch [{epoch+1}/{epoch_range}], Loss: {avg_loss:.4f}")

print(f'Finished Training, minLoss: {min_loss:.4f}')

plt.plot(Epoch, Loss)
plt.title('Training Loss')  
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()