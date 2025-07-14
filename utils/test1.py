import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.model import EmotionEfficientNet
from collections import Counter

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    class_names = ['angry', 'anxious', 'happy', 'hurt', 'neutral', 'sad', 'surprised']
    num_classes = len(class_names)

    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor()
    ])

    VAL_DIR = 'data/val'
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    print(f"✅ Loaded {len(val_dataset)} validation images.")

    model = EmotionEfficientNet(num_classes=num_classes)
    CHECKPOINT_PATH = 'efficientnet_b4_emotion_best_2.pth'
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Model loaded and ready.")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"✅ Extracted {len(all_preds)} predictions.")

    overall_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\n✅ Overall Accuracy: {overall_acc:.4f}")

    per_class_correct = Counter()
    per_class_total = Counter()

    for t, p in zip(all_labels, all_preds):
        per_class_total[t] += 1
        if t == p:
            per_class_correct[t] += 1

    print("\n✅ Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        total = per_class_total[i]
        correct = per_class_correct[i]
        acc = correct / total if total > 0 else 0.0
        print(f"  {class_name:<10}: {acc:.2%}")

    print("\n✅ Evaluation complete!")

# ------------------------
# Windows-safe entry point
# ------------------------
if __name__ == "__main__":
    main()
