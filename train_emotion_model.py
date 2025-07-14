import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.model import EmotionEfficientNet
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# 2️⃣ Data paths
train_dir = 'data/train'
val_dir = 'data/val'

# 3️⃣ Transforms
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor()
])

# 4️⃣ Datasets & Dataloaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"✅ Classes: {class_names}")

# 5️⃣ Model
model = EmotionEfficientNet(num_classes=num_classes)
checkpoint_path = 'efficientnet_b4_emotion.pth'
if os.path.exists(checkpoint_path):
    print(f"✅ Loading pretrained weights from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print("⚠️ No pretrained checkpoint found. Starting from scratch.")
model = model.to(device)

# 6️⃣ Weighted Loss
labels_in_train = [label for _, label in train_dataset]
label_counts = np.bincount(labels_in_train)
print(f"✅ Label counts: {label_counts}")

class_weights = 1.0 / torch.tensor(label_counts, dtype=torch.float)
class_weights = class_weights.to(device)
print(f"✅ Class weights: {class_weights}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# 7️⃣ Optimizer & Scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# 8️⃣ Training Loop with Best Checkpoint
num_epochs = 5
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n⭐ Epoch {epoch+1}/{num_epochs}")

    # TRAIN
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total
    print(f"✅ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # VALIDATION
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    print(f"✅ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step()

    # SAVE BEST CHECKPOINT
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'efficientnet_b4_emotion_best.pth')
        print(f"✅ Best model saved with Val Acc: {val_acc:.4f}")

print("\n✅ Training complete!")

# 9️⃣ Load Best Model for Evaluation
best_model_path = 'efficientnet_b4_emotion_best.pth'
print(f"\n✅ Loading best model from {best_model_path}")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# 10️⃣ Predict on Validation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 11️⃣ Confusion Matrix & Report
print("\n✅ Confusion Matrix and Classification Report")
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 12️⃣ Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Validation Set')
plt.tight_layout()
plt.show()

print("\n✅ All done!")
