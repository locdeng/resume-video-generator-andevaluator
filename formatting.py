import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import mediapipe as mp

# ========== CẤU HÌNH ==========
LABELS = ['angry', 'anxious', 'happy', 'neutral', 'sad', 'surprised']  # <-- Điều chỉnh đúng theo model của bạn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'emotion_resnet50_v1.0.pth'

# ========== TẠO KIẾN TRÚC MÔ HÌNH ==========
num_classes = len(LABELS)
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ========== TẢI CHECKPOINT ==========
print("🔍 Loading model checkpoint...")

try:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # Nếu là dict lớn
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        elif 'state_dict' in state:
            state = state['state_dict']

    # Xử lý nếu có 'module.' prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

model.eval().to(DEVICE)

# ========== TIỀN XỬ LÝ ẢNH ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== MEDIAPIPE DETECTOR ==========
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# ========== MỞ WEBCAM ==========
cap = cv2.VideoCapture(0)
print("✅ Webcam started. Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không đọc được frame từ webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # Clamp để không bị out of bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]
            label = "No Face"
            confidence = 0

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(face_tensor)
                    probs = F.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    label = LABELS[pred_idx]
                    confidence = probs[pred_idx].item()

                    # In xác suất từng class
                    print("\nEmotion probabilities:")
                    for i, p in enumerate(probs):
                        print(f"  {LABELS[i]:<10}: {p.item():.4f}")

            except Exception as e:
                print(f"⚠️ Lỗi xử lý face crop: {e}")
                label = "Face error"
                confidence = 0

            # VẼ LABEL LÊN FRAME
            color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Real-time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
