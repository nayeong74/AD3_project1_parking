import os
import shutil
from glob import glob

# 대상 클래스 ID들 (문자열)
target_classes = {'19', '20', '22', '28'}

# 경로 설정
src_image_dirs = [
    "/home/elicer/try1/dataset/images/train",
    "/home/elicer/try1/dataset/images/val"
]
src_label_dir = "/home/elicer/jny/labels"

dst_image_dir = "/home/elicer/jny/dataset/images"
dst_label_dir = "/home/elicer/jny/dataset/labels"
os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_label_dir, exist_ok=True)

# 전체 이미지 경로 수집
all_images = []
for d in src_image_dirs:
    all_images.extend(glob(os.path.join(d, "*.png")))

# 복사 실행
copied = 0
for img_path in all_images:
    fname = os.path.basename(img_path)
    label_name = os.path.splitext(fname)[0] + ".txt"
    label_path = os.path.join(src_label_dir, label_name)

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not any(line.strip().split()[0] in target_classes for line in lines):
            continue  # 대상 클래스 포함되지 않으면 skip

    # 복사
    shutil.copy(img_path, os.path.join(dst_image_dir, fname))
    shutil.copy(label_path, os.path.join(dst_label_dir, label_name))
    copied += 1

print(f"\n✅ 총 {copied}개 이미지와 라벨 복사 완료 (조건: 클래스 {sorted(target_classes)} 중 하나 이상 포함)")
