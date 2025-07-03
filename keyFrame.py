import os
import json

# Cấu hình
input_folder = "output"  # file keypoint JSON
trainee_id = "17"
dataset = []

# Keypoints danh sách
BODY_25_PARTS = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
    "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"
]

# Duyệt các file JSON
for file in sorted(os.listdir(input_folder)):
    if not file.endswith("_keypoints.json"):
        continue

    parts = file.split("_")
    if len(parts) < 4:
        continue

    video_id = parts[1]
    frame = parts[3]

    with open(os.path.join(input_folder, file), "r") as f:
        data = json.load(f)

    if not data["people"]:
        continue

    keypoints = data["people"][0]["pose_keypoints_2d"]
    keypoints_dict = {}

    for i, name in enumerate(BODY_25_PARTS):
        x = keypoints[i*3]
        y = keypoints[i*3+1]
        conf = keypoints[i*3+2]
        if conf > 0:
            keypoints_dict[name] = {"x": round(x), "y": round(y)}

    # Tạo 1 sample (frame)
    record = {
        "trainee": trainee_id,
        "video_id": video_id,
        "frame": frame,
        # "emotion": "중성",  # Gán cảm xúc ở đây nếu có mô hình hoặc gán tay
        "key_points": keypoints_dict
    }

    dataset.append(record)

# Ghi ra file JSON
with open("dataset_for_ai.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Completely Saved ")
