import os
import shutil
import random
from glob import glob

# 경로 설정
base_dir = '/home/elicer/jny/dataset'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'old_labels')

image_files = sorted(glob(os.path.join(image_dir, '*.png')))
random.shuffle(image_files)

# 분할 비율
split_idx = int(len(image_files) * 0.9)
train_images = image_files[:split_idx]
val_images = image_files[split_idx:]

# 대상 경로 생성
for split in ['train', 'val']:
    os.makedirs(os.path.join(image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(label_dir, split), exist_ok=True)

# 복사 함수
def move_pairs(image_list, split):
    for img_path in image_list:
        fname = os.path.basename(img_path)
        label_path = os.path.join(label_dir, fname.replace('.png', '.txt'))

        # 이동
        shutil.move(img_path, os.path.join(image_dir, split, fname))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(label_dir, split, os.path.basename(label_path)))
        else:
            print(f"❗ 라벨 없음: {label_path}")

# 실행
move_pairs(train_images, 'train')
move_pairs(val_images, 'val')

print(f"\n✅ 분할 완료: 총 {len(image_files)}개 중 {len(train_images)}개 → train, {len(val_images)}개 → val")
