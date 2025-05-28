import os
from collections import defaultdict

label_dir = "dataset/labels_segformer/val"
image_dir = "/home/elicer/dataset/images/val"

class_counts = defaultdict(int)
total_objects = 0
total_labels = 0

# 전체 라벨 파일 순회
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        total_labels += 1
        with open(os.path.join(label_dir, label_file), "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue  # 빈 줄 무시
                try:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
                except ValueError:
                    print(f"❗ 잘못된 라벨 → 무시됨: {label_file} / '{line.strip()}'")
                    continue

# 이미지 수
image_count = len([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"✅ 전체 이미지 수: {image_count}")
print(f"✅ 전체 라벨(.txt) 수: {total_labels}")
print(f"✅ 전체 객체 수: {total_objects}")
print(f"✅ 클래스별 객체 수:")
for cls, count in sorted(class_counts.items()):
    print(f"  클래스 {cls}: {count}개")
