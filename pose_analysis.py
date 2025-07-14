import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

# ----------------------------------------
# ⭐ Helper: Extract landmarks in pixel
# ----------------------------------------
def extract_landmarks(landmarks, image_shape):
    h, w = image_shape[:2]
    coords = []
    for lm in landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        coords.append([x_px, y_px])
    return coords


# ----------------------------------------
# ⭐ Pose Analysis per Frame
# ----------------------------------------
def analyze_interview_pose(landmarks, image_shape, prev_wrist_coords=None):
    h, w = image_shape[:2]
    result = {}
    idx = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_wrist": 15,
        "right_wrist": 16
    }

    # ⭐ Posture detection
    avg_shoulder_y = (landmarks[idx["left_shoulder"]][1] + landmarks[idx["right_shoulder"]][1]) / 2
    avg_hip_y = (landmarks[idx["left_hip"]][1] + landmarks[idx["right_hip"]][1]) / 2
    vertical_dist = avg_hip_y - avg_shoulder_y
    if vertical_dist > h * 0.4:
        result["posture"] = "standing"
    elif vertical_dist < h * 0.2:
        result["posture"] = "sitting"
    else:
        result["posture"] = "unknown"

    # ⭐ Shoulder balance
    shoulder_diff = abs(landmarks[idx["left_shoulder"]][1] - landmarks[idx["right_shoulder"]][1])
    if shoulder_diff < h * 0.02:
        result["shoulder_balance"] = "aligned"
    elif landmarks[idx["left_shoulder"]][1] < landmarks[idx["right_shoulder"]][1]:
        result["shoulder_balance"] = "left_high"
    else:
        result["shoulder_balance"] = "right_high"

    # ⭐ Arms openness
    left_wrist_x = landmarks[idx["left_wrist"]][0]
    right_wrist_x = landmarks[idx["right_wrist"]][0]
    shoulder_width = abs(landmarks[idx["left_shoulder"]][0] - landmarks[idx["right_shoulder"]][0])
    wrists_span = abs(right_wrist_x - left_wrist_x)
    if wrists_span > shoulder_width * 1.5:
        result["arms_position"] = "open"
    else:
        result["arms_position"] = "closed"

    # ⭐ Motion energy
    # if prev_wrist_coords:
    #     prev_left, prev_right = prev_wrist_coords
    #     motion_left = np.linalg.norm(np.array([left_wrist_x, landmarks[idx["left_wrist"]][1]]) - np.array(prev_left))
    #     motion_right = np.linalg.norm(np.array([right_wrist_x, landmarks[idx["right_wrist"]][1]]) - np.array(prev_right))
    #     avg_motion = (motion_left + motion_right) / 2
    #     result["motion_energy"] = "active" if avg_motion > 15 else "still"
    # else:
    #     result["motion_energy"] = "unknown"

    # return result


# ----------------------------------------
# ⭐ Hand Analysis per Frame
# ----------------------------------------
def analyze_hand_gesture(landmarks, prev_landmarks=None):
    WRIST = 0
    TIP_IDS = [4, 8, 12, 16, 20]
    wrist_xy = np.array(landmarks[WRIST])
    tips = [np.array(landmarks[i]) for i in TIP_IDS]

    dists = [np.linalg.norm(tip - wrist_xy) for tip in tips]
    avg_dist = np.mean(dists)
    gesture = "open" if avg_dist > 50 else "closed"

    result = {
        "gesture": gesture,
        "avg_finger_distance": avg_dist
    }

    # if prev_landmarks:
    #     motions = [
    #         np.linalg.norm(np.array(lm) - np.array(plm))
    #         for lm, plm in zip(landmarks, prev_landmarks)
    #     ]
    #     avg_motion = np.mean(motions)
    #     result["motion_energy"] = "active" if avg_motion > 10 else "still"
    # else:
    #     result["motion_energy"] = "unknown"

    return result


# ----------------------------------------
# ⭐ Aggregation Helpers
# ----------------------------------------
def get_pose_counts(pose_analysis_list):
    counters = {
        "posture": Counter(),
        "shoulder_balance": Counter(),
        "arms_position": Counter(),
        # "motion_energy": Counter()
    }
    for item in pose_analysis_list:
        if not item:
            continue
        for key in counters.keys():
            counters[key][item.get(key, "unknown")] += 1
    return counters

def get_hand_counts(hand_analysis_list):
    counters = {
        "gesture": Counter(),
        # "motion_energy": Counter()
    }
    for item in hand_analysis_list:
        for key in counters.keys():
            counters[key][item.get(key, "unknown")] += 1
    return counters


# ----------------------------------------
# ⭐ Charts
# ----------------------------------------
def render_pose_charts(pose_counters, container):
    with container.expander("📊 Pose Charts", expanded=False):
        features = list(pose_counters.keys())
        for i in range(0, len(features), 2):
            cols = st.columns(2)
            for idx in range(2):
                if i + idx < len(features):
                    feature = features[i + idx]
                    counter = pose_counters[feature]
                    if counter:
                        with cols[idx]:
                            st.markdown(f"**{feature.capitalize()}**")
                            df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                            st.bar_chart(df)

def render_hand_charts(hand_counters, container):
    with container.expander("📊 Hand Charts", expanded=False):
        features = list(hand_counters.keys())
        cols = st.columns(len(features))
        for idx, feature in enumerate(features):
            counter = hand_counters[feature]
            if counter:
                with cols[idx]:
                    st.markdown(f"**{feature.capitalize()}**")
                    df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                    st.bar_chart(df)


# ----------------------------------------
# ⭐ Emotion Charts (Insight)
# ----------------------------------------
def render_charts_and_insights(container):
    if len(st.session_state["emotion_labels"]) < 1:
        container.info("📊 아직 기록된 감정 데이터가 없습니다. Start를 눌러 분석을 시작해 보세요!")
        return

    df = pd.DataFrame({"emotion": st.session_state["emotion_labels"]})
    counts = df["emotion"].value_counts().sort_index()
    total_frames = len(st.session_state["emotion_labels"])
    most_common = df["emotion"].mode()[0]

    with container.expander("😊 Emotion Summary", expanded=False):
        st.subheader("📝 인터뷰 세션 감성 분석")
        st.markdown(f"""
        - **가장 많이 탐지된 감정**: {most_common}
        - **총 기록 수**: {total_frames} 프레임
        - **감정 비율**:
        """)
        for emo, cnt in counts.items():
            pct = 100 * cnt / total_frames
            st.markdown(f"  - {emo}: {pct:.1f}%")

        st.markdown("---")
        st.subheader("📊 감정 분포")
        st.bar_chart(counts)


# ----------------------------------------
# ⭐ Session Summary
# ----------------------------------------
def summarize_session(emotion_labels, pose_analysis_list, hand_analysis_list, face_analysis_list, st_container):
    st_container.subheader("📝 인터뷰 세션 분석 요약")

    # 1️⃣ Emotion
    if emotion_labels:
        render_charts_and_insights(st_container)

    # 2️⃣ Pose
    if pose_analysis_list:
        pose_counters = get_pose_counts(pose_analysis_list)
        with st_container.expander("🧍‍♂️ Pose Summary", expanded=False):
            st.markdown("**📌 자세 분석 카운트 및 차트:**")
            for feature, counter in pose_counters.items():
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

                if counter:
                    st.markdown(f"**📊 {feature.capitalize()} 차트**")
                    df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                    st.bar_chart(df)

    # 3️⃣ Hand
    if hand_analysis_list:
        hand_counters = get_hand_counts(hand_analysis_list)
        with st_container.expander("✋ Hand Summary", expanded=False):
            st.markdown("**📌 손 제스처 분석 카운트 및 차트:**")
            for feature, counter in hand_counters.items():
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

                if counter:
                    st.markdown(f"**📊 {feature.capitalize()} 차트**")
                    df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                    st.bar_chart(df)

    # 4️⃣ Face
    if face_analysis_list:
        face_counters = {
            "eye_state": Counter(),
            "mouth_state": Counter(),
            "head_tilt": Counter()
        }

        for item in face_analysis_list:
            for key in face_counters.keys():
                face_counters[key][item.get(key, "unknown")] += 1

        with st_container.expander("😊 Face Summary", expanded=False):
            st.markdown("**📌 얼굴 분석 카운트 및 차트:**")
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

                if counter:
                    st.markdown(f"**📊 {feature.capitalize()} 차트**")
                    df = pd.DataFrame.from_dict(counter, orient='index', columns=['count'])
                    st.bar_chart(df)

