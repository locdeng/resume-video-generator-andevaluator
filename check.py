import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# ==== CONFIG ====
INPUT_FOLDER = 'dataset/test/img'
MODEL_PATH = 'emotion_classifier_7class.pth'
OUTPUT_JSON = 'predicted_labels.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Classes (chỉnh đúng theo train) ====
class_names = ['angry', 'anxious', 'happy', 'hurt', 'neutral', 'sad', 'surprised']

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==== Load model ====
NUM_CLASSES = len(class_names)
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# ==== Predict all images ====
results = []
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in tqdm(image_files, desc="Predicting"):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ Skip {img_name}: {e}")
        continue

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        predicted_label = class_names[pred.item()]

    results.append({
        "image": img_name,
        "predicted_label": predicted_label
    })

# ==== Save results ====
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Done! Predictions saved to {OUTPUT_JSON}")
