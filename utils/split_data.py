import os
import random
import shutil

# 1️⃣ Đường dẫn thư mục
SOURCE_DIR = 'data/all'     # thư mục ảnh ban đầu (chia theo nhãn)
TRAIN_DIR = 'data/train'    # thư mục muốn lưu train
VAL_DIR = 'data/val'        # thư mục muốn lưu val

VAL_RATIO = 0.2                 # ví dụ 20% val

# 2️⃣ Tạo thư mục đích
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

make_dir(TRAIN_DIR)
make_dir(VAL_DIR)

# 3️⃣ Lặp qua từng class
for class_name in os.listdir(SOURCE_DIR):
    class_src_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_src_path):
        continue

    images = os.listdir(class_src_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    # Chia train/val
    n_total = len(images)
    n_val = int(n_total * VAL_RATIO)
    val_images = images[:n_val]
    train_images = images[n_val:]

    print(f"Class: {class_name} | Total: {n_total} | Train: {len(train_images)} | Val: {len(val_images)}")

    # Tạo thư mục class trong train/val
    class_train_dir = os.path.join(TRAIN_DIR, class_name)
    class_val_dir = os.path.join(VAL_DIR, class_name)
    make_dir(class_train_dir)
    make_dir(class_val_dir)

    # Copy ảnh
    for img in train_images:
        src = os.path.join(class_src_path, img)
        dst = os.path.join(class_train_dir, img)
        shutil.copyfile(src, dst)

    for img in val_images:
        src = os.path.join(class_src_path, img)
        dst = os.path.join(class_val_dir, img)
        shutil.copyfile(src, dst)

print("✅ Done splitting dataset!")
