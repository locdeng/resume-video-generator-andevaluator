import cv2
import mediapipe as mp
from deepface import DeepFace

# ========== Mediapipe init ==========
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# ========== Webcam ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được webcam!")
    exit()

print("✅ Webcam started. Nhấn Q để thoát.")

# ========== Mediapipe Holistic ==========
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được frame.")
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Holistic
        results = holistic.process(frame_rgb)

        # Copy frame to draw
        annotated_frame = frame.copy()

        # ========== DRAW POSE ==========
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # ========== DRAW HANDS ==========
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # ========== FACE CROP FOR EMOTION ==========
        if results.face_landmarks:
            # Extract rough face bounding box from 468 landmarks
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in results.face_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in results.face_landmarks.landmark]

            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

            face_roi = frame[y_min:y_max, x_min:x_max]

            try:
                if face_roi.size > 0:
                    # DeepFace analyze
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = result[0]['dominant_emotion']
                    emotion_score = result[0]['emotion'][dominant_emotion]

                    # Draw bounding box and label
                    label = f"{dominant_emotion.capitalize()} ({emotion_score:.1f}%)"
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"⚠️ DeepFace error: {e}")

        # ========== Show result ==========
        cv2.imshow("Holistic + DeepFace Emotion", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
