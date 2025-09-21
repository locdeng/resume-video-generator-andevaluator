
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
from collections import Counter
from utils.gemini_utils import build_insight_prompt_ko
from pose_analysis import *
from about_me_gen import generate_text_gemini

def generate_llm_insight():
    st.subheader("🪄 자동 생성 인사이트")

    # Count raw
    from collections import Counter
    pose_counts = get_pose_counts(st.session_state.get("pose_analysis", []))
    hand_counts = get_hand_counts(st.session_state.get("hand_analysis", []))
    emotion_counts = Counter(st.session_state.get("emotion_labels", []))
    face_counts = get_face_counts(st.session_state.get("face_analysis",[]))

    # Build prompt
    prompt = build_insight_prompt_ko(pose_counts, hand_counts, emotion_counts,face_counts)

    # 👉 Optionally: Call LLM
    response =  generate_text_gemini(prompt)
    st.markdown("✅Insight:")
    st.markdown(response)

# ----------------------------------------
# ⭐ Helper: Bỏ prefix key
# ----------------------------------------
def remove_prefix(state_dict, prefix='model.'):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


# ----------------------------------------
# ⭐ Face landmarks -> np array
# ----------------------------------------
def landmark_to_np(landmarks, image_shape):
    h, w = image_shape[:2]
    return np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])


# ----------------------------------------
# ⭐ Eye Aspect Ratio
# ----------------------------------------
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def extract_eye_ratio(landmarks_px):
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    left_eye = landmarks_px[LEFT_EYE_IDX]
    right_eye = landmarks_px[RIGHT_EYE_IDX]
    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)
    return left_EAR, right_EAR


# ----------------------------------------
# ⭐ Mouth Aspect Ratio
# ----------------------------------------
def mouth_aspect_ratio(landmarks_px):
    top = landmarks_px[13]
    bottom = landmarks_px[14]
    left = landmarks_px[78]
    right = landmarks_px[308]
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / horizontal


# ----------------------------------------
# ⭐ Head Roll and Pitch
# ----------------------------------------
def estimate_roll_angle(landmarks_px):
    left_eye = landmarks_px[33]
    right_eye = landmarks_px[263]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle_rad = np.arctan2(dy, dx)
    return np.degrees(angle_rad)

def estimate_pitch_proxy(landmarks_px):
    nose = landmarks_px[1]
    chin = landmarks_px[152]
    return np.linalg.norm(nose - chin)


# ----------------------------------------
# ⭐ Analyze Face Features
# ----------------------------------------
def analyze_face_features(landmarks_px):
    left_EAR, right_EAR = extract_eye_ratio(landmarks_px)
    mouth_ratio = mouth_aspect_ratio(landmarks_px)
    roll_angle = estimate_roll_angle(landmarks_px)
    pitch_proxy = estimate_pitch_proxy(landmarks_px)
    avg_eye = (left_EAR + right_EAR) / 2

    return {
        "left_eye_EAR": left_EAR,
        "right_eye_EAR": right_EAR,
        "avg_eye_EAR": avg_eye,
        "mouth_ratio": mouth_ratio,
        "roll_angle_deg": roll_angle,
        "pitch_proxy_dist": pitch_proxy,
        "eye_state": "closed" if avg_eye < 0.2 else "open",
        "mouth_state": "open" if mouth_ratio > 0.5 else "closed",
        "head_tilt": "tilted" if abs(roll_angle) > 10 else "straight"
    }


# ----------------------------------------
# ⭐ Preprocessing
# ----------------------------------------
def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_face_image(pil_image, gamma_value=1.2):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    img_eq = adjust_gamma(img_eq, gamma=gamma_value)
    img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ----------------------------------------
# ⭐ Webcam Loop
# ----------------------------------------
def realtime_capture_loop(model, holistic, transform, EMOTION_LABELS, video_container):
    st.info("✅ 웹캠이 실행 중입니다. Stop을 누르면 종료됩니다.")
    stframe = video_container
    prev_wrist_coords = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ 웹캠을 열 수 없습니다.")
        return

    while st.session_state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(frame_rgb)

        # ⭐ Pose
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS
            )
            from pose_analysis import extract_landmarks, analyze_interview_pose
            landmarks_px = extract_landmarks(results.pose_landmarks.landmark, frame.shape)
            pose_result = analyze_interview_pose(landmarks_px, frame.shape, prev_wrist_coords)
            st.session_state["pose_analysis"].append(pose_result)
            prev_wrist_coords = (landmarks_px[15], landmarks_px[16])

        # ⭐ Hands
        from pose_analysis import analyze_hand_gesture
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS
                )
                landmarks_px = extract_landmarks(hand_landmarks.landmark, frame.shape)
                hand_result = analyze_hand_gesture(landmarks_px)
                st.session_state["hand_analysis"].append(hand_result)

        # ⭐ Face - Emotion detection
        if results.face_landmarks:
            ih, iw, _ = frame.shape
            face_coords = np.array([
                [int(lm.x * iw), int(lm.y * ih)] for lm in results.face_landmarks.landmark
            ])
            x_min, y_min = np.min(face_coords[:,0]), np.min(face_coords[:,1])
            x_max, y_max = np.max(face_coords[:,0]), np.max(face_coords[:,1])
            x1, y1 = max(0, x_min - 20), max(0, y_min - 20)
            x2, y2 = min(iw, x_max + 20), min(ih, y_max + 20)

            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.shape[0] > 30 and face_crop.shape[1] > 30:
                face_pil = Image.fromarray(face_crop)
                face_pil = preprocess_face_image(face_pil)
                input_tensor = transform(face_pil).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = EMOTION_LABELS[pred]
                st.session_state["emotion_labels"].append(label)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ⭐ Face landmarks ➜ feature extraction
            landmarks_px = landmark_to_np(results.face_landmarks.landmark, frame.shape)
            face_result = analyze_face_features(landmarks_px)
            st.session_state["face_analysis"].append(face_result)

        # ⭐ Show
        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ 스트리밍이 종료되었습니다. 아래에서 결과를 확인하세요.")


# ----------------------------------------
# ⭐ Main
# ----------------------------------------
def run_realtime_analysis():
    
    st.markdown("""
        <style>
        
         @font-face {
            font-family: 'SB_B';
            src: url('assets/fonts/SF.ttf') format('truetype');
        }
        
                /* Toàn bộ trang (nền đen) */
        html, body {
            background-color: #f0e8db !important;
            font-family: 'SF',sans-serif;
        }

        /* Nền vùng nội dung */
        [data-testid="stAppViewContainer"] {
            background-color: #f0e8db !important;
        }

        /* Nền container chính */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #f0e8db !important;
            padding: 0rem 1rem; /* giảm padding nếu muốn */
            max-width: 100% !important;  /* full width */
        }

        /* Optional: Sidebar cũng nền đen */
        [data-testid="stSidebar"] {
            background-color: #77C9D4 !important;
        }
        .intro-title {
            font-size: 48px;
            font-weight: 800;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: 30px;
        }
        .intro-sub {
            font-size: 18px;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .feature-box {
            background: #F2EFE7 ;
            padding: 30px;
            border-radius: 15px;
            margin: 10px 20px;
            color: #2b2b2b;
            border: 2px solid white;
            font-family: 'SF',sans-serif;
            text-align: center;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            font-family: 'SF',sans-serif;
            color: #2b2b2b;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            background-color: #F2EFE7;
            border: 2px solid white;
            color: #2b2b2b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intro-title">실시간 영상 분석 (Webcam)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ✅ Mediapipe Holistic (Face+Pose+Hands)  
    ✅ 감정 분류 (EfficientNet)  
    ✅ 자세/손 제스처/얼굴표정 분석 결과 제공
    """)

    if "emotion_labels" not in st.session_state:
        st.session_state["emotion_labels"] = []
    if "pose_analysis" not in st.session_state:
        st.session_state["pose_analysis"] = []
    if "hand_analysis" not in st.session_state:
        st.session_state["hand_analysis"] = []
    if "face_analysis" not in st.session_state:
        st.session_state["face_analysis"] = []
    if "running" not in st.session_state:
        st.session_state["running"] = False

    cols = st.columns(3)
    with cols[0]:
        start_button = st.button("▶️ 웹캠 시작")
    with cols[1]:
        stop_button = st.button("⏹️ 웹캠 종료")
    with cols[2]:
        reset_button = st.button("🗑️ 데이터 초기화")

    if start_button:
        st.session_state["running"] = True
    if stop_button:
        st.session_state["running"] = False
    if reset_button:
        st.session_state["emotion_labels"] = []
        st.session_state["pose_analysis"] = []
        st.session_state["hand_analysis"] = []
        st.session_state["face_analysis"] = []
        st.success("✅ 데이터가 초기화되었습니다!")

    video_container = st.empty()
    insight_container = st.container()

    EMOTION_LABELS = ['angry', 'anxious', 'happy', 'hurt', 'neutral', 'sad', 'surprised']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_name("efficientnet-b4")
    model._fc = torch.nn.Linear(model._fc.in_features, len(EMOTION_LABELS))
    checkpoint = torch.load("efficientnet_b4_emotion_best.pth", map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = remove_prefix(state_dict, prefix='model.')
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    model.device = DEVICE

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if st.session_state["running"]:
        realtime_capture_loop(
            model, holistic, transform, EMOTION_LABELS,
            video_container
        )

    if not st.session_state["running"]:

        if st.session_state["pose_analysis"] or st.session_state["hand_analysis"]:
            from pose_analysis import summarize_session
            
            summarize_session(
                st.session_state["emotion_labels"],
                st.session_state["pose_analysis"],
                st.session_state["hand_analysis"],
                st.session_state["face_analysis"],
                insight_container
            )

        
    if insight_container.button("🪄 인사이트 생성"):
        generate_llm_insight()
        
def run_uploaded_video_analysis():
    
    st.markdown("""
        <style>
        
         @font-face {
            font-family: 'SB_B';
            src: url('assets/fonts/SF.ttf') format('truetype');
        }
        
                /* Toàn bộ trang (nền đen) */
        html, body {
            background-color: #f0e8db !important;
            font-family: 'SF',sans-serif;
        }

        /* Nền vùng nội dung */
        [data-testid="stAppViewContainer"] {
            background-color: #f0e8db !important;
        }

        /* Nền container chính */
        [data-testid="stAppViewBlockContainer"] {
            background-color: #f0e8db !important;
            padding: 0rem 1rem; /* giảm padding nếu muốn */
            max-width: 100% !important;  /* full width */
        }

        /* Optional: Sidebar nếu bạn muốn cũng nền đen */
        [data-testid="stSidebar"] {
            background-color: #77C9D4 !important;
        }
        .intro-title {
            font-size: 48px;
            font-weight: 800;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: 30px;
        }
        .intro-sub {
            font-size: 18px;
            color: #2b2b2b;
            text-align: center;
            font-family: 'SF',sans-serif;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .feature-box {
            background: #F2EFE7 ;
            padding: 30px;
            border-radius: 15px;
            margin: 10px 20px;
            color: #2b2b2b;
            border: 2px solid white;
            font-family: 'SF',sans-serif;
            text-align: center;
        }
        .feature-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
            font-family: 'SF',sans-serif;
            color: #2b2b2b;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            background-color: #F2EFE7;
            border: 2px solid white;
            color: #2b2b2b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intro-title">업로드 영상 분석</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ✅ 업로드한 영상 파일을 프레임별로 분석합니다.  
    ✅ Mediapipe Holistic으로 자세/손 제스처/얼굴 감정 분석을 수행합니다.  
    ✅ 분석 후 인사이트 요약을 제공합니다.
    """)

    # ⭐ Session state 초기화
    if "emotion_labels" not in st.session_state:
        st.session_state["emotion_labels"] = []
    if "emotion_counts" not in st.session_state:
        st.session_state["emotion_counts"] = {}
    if "pose_analysis" not in st.session_state:
        st.session_state["pose_analysis"] = []
    if "hand_analysis" not in st.session_state:
        st.session_state["hand_analysis"] = []
    if "face_analysis" not in st.session_state:
        st.session_state["face_analysis"] = []

    # ⭐ 업로드 위젯
    uploaded_file = st.file_uploader("📤 분석할 비디오 업로드", type=["mp4", "avi", "mov"])
    if uploaded_file is None:
        st.info("✅ 분석할 영상을 업로드해주세요.")
        return
    
    # ⭐ 비디오 저장
    video_path = "temp_uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ 비디오 업로드 완료!")

    # ⭐ 분석 시작 버튼
    if st.button("▶️ 업로드된 비디오 분석 시작"):
        st.session_state["running"] = True

    # ⭐ 분석 루프
    if st.session_state.get("running", False):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EMOTION_LABELS = ['angry', 'anxious', 'happy', 'hurt', 'neutral', 'sad', 'surprised']

        # Load model
        model = EfficientNet.from_name("efficientnet-b4")
        model._fc = torch.nn.Linear(model._fc.in_features, len(EMOTION_LABELS))
        checkpoint = torch.load("efficientnet_b4_emotion_best.pth", map_location=DEVICE)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = remove_prefix(state_dict, prefix='model.')
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()
        model.device = DEVICE

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ⭐ 실제 분석 수행
        analyze_uploaded_video(video_path, model, holistic, transform, EMOTION_LABELS)

        st.session_state["running"] = False
    insight_container = st.container()
    # ⭐ 결과 요약
    if not st.session_state.get("running", False):
        if (st.session_state.get("emotion_labels") or 
            st.session_state.get("pose_analysis") or 
            st.session_state.get("hand_analysis") or
            st.session_state.get("face_analysis")):
            summarize_session(
                st.session_state["emotion_labels"],
                st.session_state["pose_analysis"],
                st.session_state["hand_analysis"],
                st.session_state["face_analysis"],
                st.container()
            )
    if insight_container.button("🪄 인사이트 생성"):
        generate_llm_insight()
        
        
def analyze_uploaded_video(video_path, model, holistic, transform, EMOTION_LABELS):
    st.info("✅ 업로드된 비디오를 분석 중입니다...")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_wrist_coords = None
    insight_container = st.container()
    # ⭐ 프레임 출력할 공간
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ⭐ Mediapipe Holistic 분석
        results = holistic.process(frame_rgb)

        # ⭐ Pose
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS
            )
            landmarks_px = extract_landmarks(results.pose_landmarks.landmark, frame.shape)
            pose_result = analyze_interview_pose(landmarks_px, frame.shape, prev_wrist_coords)
            st.session_state["pose_analysis"].append(pose_result)
            prev_wrist_coords = (landmarks_px[15], landmarks_px[16])

        # ⭐ Hands
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS
                )
                landmarks_px = extract_landmarks(hand_landmarks.landmark, frame.shape)
                hand_result = analyze_hand_gesture(landmarks_px)
                st.session_state["hand_analysis"].append(hand_result)

        # ⭐ Face - Emotion
        if results.face_landmarks:
            ih, iw, _ = frame.shape
            face_coords = np.array([
                [int(lm.x * iw), int(lm.y * ih)] for lm in results.face_landmarks.landmark
            ])
            x_min = np.min(face_coords[:,0])
            y_min = np.min(face_coords[:,1])
            x_max = np.max(face_coords[:,0])
            y_max = np.max(face_coords[:,1])

            x1, y1 = max(0, x_min - 20), max(0, y_min - 20)
            x2, y2 = min(iw, x_max + 20), min(ih, y_max + 20)

            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.shape[0] > 30 and face_crop.shape[1] > 30:
                face_pil = Image.fromarray(face_crop)
                face_pil = preprocess_face_image(face_pil)
                input_tensor = transform(face_pil).unsqueeze(0).to(model.device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item()
                    label = EMOTION_LABELS[pred]

                st.session_state["emotion_labels"].append(label)
                st.session_state["emotion_counts"][label] = st.session_state["emotion_counts"].get(label, 0) + 1

                # ⭐ 얼굴 영역 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ⭐ Face landmarks ➜ feature extraction
            landmarks_px = landmark_to_np(results.face_landmarks.landmark, frame.shape)
            face_result = analyze_face_features(landmarks_px)
            st.session_state["face_analysis"].append(face_result)

        # ⭐ ⭐ 스트림릿 화면에 현재 프레임 표시 ⭐ ⭐
        frame_placeholder.image(frame, channels="BGR")

        frame_count += 1

    cap.release()
    st.success("✅ 비디오 분석이 완료되었습니다!")
    
    if insight_container.button("🪄 인사이트 생성"):
        generate_llm_insight()
