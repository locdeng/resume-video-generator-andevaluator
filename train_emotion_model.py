import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#cuo
# ===== Info =====
DATA_DIR = 'dataset/train'
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Sử dụng thiết bị:", DEVICE)

# ===== Image Preprocessing =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# ===== Load =====
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Lớp cảm xúc:", train_dataset.classes)

# ===== MÔ HÌNH =====
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ===== LOSS & OPTIMIZER =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===== Traning =====
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f" Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} - Accuracy: {acc:.4f}")

# ===== LƯU MÔ HÌNH =====
torch.save(model.state_dict(), "emotion_classifier_7class.pth")
print(" Successfully Saved: emotion_classifier_7class.pth")