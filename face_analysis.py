import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

# ----------------------------------------
# ⭐ Landmark chuyển thành np array (pixels)
# ----------------------------------------
def landmark_to_np(landmarks, image_shape):
    h, w = image_shape[:2]
    return np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

# ----------------------------------------
# ⭐ Eye Aspect Ratio (EAR)
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
# ⭐ Mouth Aspect Ratio (MAR)
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
# ⭐ Head Roll (Tilt Angle)
# ----------------------------------------
def estimate_roll_angle(landmarks_px):
    left_eye = landmarks_px[33]
    right_eye = landmarks_px[263]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle_rad = np.arctan2(dy, dx)
    return np.degrees(angle_rad)

# ----------------------------------------
# ⭐ Head Pitch Proxy (Vertical Distance)
# ----------------------------------------
def estimate_pitch_proxy(landmarks_px):
    nose = landmarks_px[1]
    chin = landmarks_px[152]
    vertical = np.linalg.norm(nose - chin)
    return vertical

# ----------------------------------------
# ⭐ Phân tích Face 1 frame
# ----------------------------------------
def analyze_face_features(landmarks_px):
    left_EAR, right_EAR = extract_eye_ratio(landmarks_px)
    mouth_ratio = mouth_aspect_ratio(landmarks_px)
    roll_angle = estimate_roll_angle(landmarks_px)
    pitch_proxy = estimate_pitch_proxy(landmarks_px)

    result = {
        "left_eye_EAR": left_EAR,
        "right_eye_EAR": right_EAR,
        "avg_eye_EAR": (left_EAR + right_EAR) / 2,
        "mouth_ratio": mouth_ratio,
        "roll_angle_deg": roll_angle,
        "pitch_proxy_dist": pitch_proxy,
        # Threshold-based labels
        "eye_state": "closed" if (left_EAR + right_EAR)/2 < 0.2 else "open",
        "mouth_state": "open" if mouth_ratio > 0.5 else "closed",
        "head_tilt": "tilted" if abs(roll_angle) > 10 else "straight"
    }
    return result

# ----------------------------------------
# ⭐ Aggregate session stats
# ----------------------------------------
def get_face_counts(face_analysis_list):
    counters = {
        "eye_state": Counter(),
        "mouth_state": Counter(),
        "head_tilt": Counter()
    }
    for item in face_analysis_list:
        for key in counters.keys():
            counters[key][item.get(key, "unknown")] += 1
    return counters

# ----------------------------------------
# ⭐ Render charts
# ----------------------------------------
def render_face_charts(face_counters, container):
    with container.expander("📊 Face Charts", expanded=False):
        features = list(face_counters.keys())
        for i in range(0, len(features), 2):
            cols = st.columns(2)
            for idx in range(2):
                if i + idx < len(features):
                    feature = features[i + idx]
                    counter = face_counters[feature]
                    if counter:
                        with cols[idx]:
                            st.markdown(f"**{feature.capitalize()}**")
                            df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                            st.bar_chart(df)

# ----------------------------------------
# ⭐ Summary + Insight
# ----------------------------------------
def summarize_face_session(face_analysis_list, st_container):
    st_container.subheader("📝 얼굴 표정/포즈 분석 요약")

    if not face_analysis_list:
        st_container.info("❗ 얼굴 데이터가 없습니다.")
        return

    face_counters = get_face_counts(face_analysis_list)

    with st_container.expander("😊 Face Summary", expanded=False):
        st.markdown("**📌 얼굴 분석 카운트:**")
        for feature, counter in face_counters.items():
            total_frames = sum(counter.values())
            if total_frames == 0:
                continue
            most_common = counter.most_common(1)[0][0]
            st.markdown(f"**✅ {feature.capitalize()} 분석:**")
            st.markdown(f"- 가장 많이 탐지된 상태: **{most_common}**")
            st.markdown(f"- 총 기록 수: **{total_frames} 프레임**")
            st.markdown(f"- 상태 비율:")
            for k, v in counter.items():
                pct = 100 * v / total_frames
                st.markdown(f"  - {k}: {pct:.1f}%")
            st.markdown("---")

        # Charts
        render_face_charts(face_counters, st_container)
