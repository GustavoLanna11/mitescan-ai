import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.cnn import MiteScanCNN
import torch.nn as nn
import torch.optim as optim

# transformação
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# dataset
train_data = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# modelo
model = MiteScanCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# treino
for epoch in range(20):
    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} finalizada")

# salvar
torch.save(model.state_dict(), "model.pth")