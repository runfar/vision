import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 128
    # num_workers=0 avoids multiprocessing issues; increase after confirming main-guard works
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, 512),
                nn.Tanh(),
                nn.Linear(512, 10)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 50
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        acc = correct / len(test_loader.dataset) * 100
        print(f"Epoch {epoch:2d}/{epochs} - loss: {epoch_loss:.4f} - val_acc: {acc:.2f}%")

    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    print("정확률:", correct / len(test_loader.dataset) * 100)

if __name__ == '__main__':
    main()