import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from utils.model import EmotionEfficientNet   # copy từ repo

# 1️⃣ Thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2️⃣ Classes
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3️⃣ Transform (phải match training!)
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor()
])

# 4️⃣ Dataset riêng của bạn
data_dir = 'data/val'  # <<< chỉnh đúng path!
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
print(f"Total images: {len(dataset)}")

# 5️⃣ Load model.pth
num_classes = len(class_names)
model = EmotionEfficientNet(num_classes=num_classes)
model.load_state_dict(torch.load('efficientnet_b4_emotion_best_2.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model.pth!")

# 6️⃣ Predict
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Prediction done!")

# 7️⃣ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# 8️⃣ Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 9️⃣ Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on My Data')
plt.tight_layout()
plt.show()
