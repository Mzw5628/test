import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import recognizer


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model = recognizer()
model.load_state_dict(torch.load('model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    train_dataset = datasets.ImageFolder(root=r'dataset\training_set\training_set', transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 5
    min_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), 'model.pth')
            print("Model saved")

def test():
    model.eval()
    test_dataset = datasets.ImageFolder(root=r'dataset\test_set\test_set', transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train()
    test()