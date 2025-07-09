import json

# ===== CONFIG =====
GROUND_TRUTH_FILE = 'merged_label_data.json'
PREDICTIONS_FILE = 'predicted_labels.json'
OUTPUT_FILE = 'comparison_results.json'

# ===== Load Files =====
with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
    pred_data = json.load(f)

# ===== Normalize Filenames =====
def clean_name(name):
    return name.strip().lower()

pred_dict = {clean_name(item['image']): item['predicted_label'] for item in pred_data}
gt_dict = {clean_name(item['image']): item['true_label'] for item in gt_data}

# ===== Compare =====
correct = 0
total = 0
missing_in_pred = 0
details = []

for img_name in gt_dict:
    clean_img = clean_name(img_name)
    true_label = gt_dict[clean_img]
    pred_label = pred_dict.get(clean_img)

    if pred_label is None:
        missing_in_pred += 1
        details.append({
            "image": img_name,
            "true_label": true_label,
            "predicted_label": None,
            "match": False,
            "reason": "Not found in predictions"
        })
        continue

    match = (true_label == pred_label)
    if match:
        correct += 1
    total += 1

    details.append({
        "image": img_name,
        "true_label": true_label,
        "predicted_label": pred_label,
        "match": match
    })

accuracy = correct / total if total > 0 else 0.0

# ===== Save Results =====
result_summary = {
    "total_compared": total,
    "correct": correct,
    "accuracy": accuracy,
    "missing_in_predictions": missing_in_pred,
    "details": details
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=4)

print(f"✅ Done! Accuracy: {accuracy:.4f}")
print(f"✅ Total images compared: {total}")
print(f"✅ Missing in predictions: {missing_in_pred}")
print(f"✅ Results saved to {OUTPUT_FILE}")
