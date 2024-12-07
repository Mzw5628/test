import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from model import Net
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("./data", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

test_loader = get_data_loader(False)

correct = 0
total = 0


net = Net()
net.load_state_dict(torch.load("parameters/model.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)


net.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")

dataiter = iter(test_loader)
images, labels = next(dataiter)

index = random.randint(0, len(images) - 1) 
image = images[index]
label = labels[index].item()

image_tensor = image.unsqueeze(0).to(device)
with torch.no_grad():
    output = net(image_tensor)
    _, predicted = torch.max(output, 1)

predicted_label = predicted.item()

plt.imshow(image.squeeze(), cmap='gray')
title_text = f'Predicted: {predicted_label}, Actual: {label}'
if predicted_label == label:
    title_text += ' (Correct)'
else:
    title_text += ' (Incorrect)'
plt.title(title_text)
plt.axis('off')
plt.show()