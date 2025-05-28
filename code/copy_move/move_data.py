import os
import shutil
from glob import glob

# 기존 경로
src_img_dirs = [
    "/home/elicer/try1/dataset/images/train",
    "/home/elicer/try1/dataset/images/val"
]
src_lbl_dirs = [
    "/home/elicer/try1/dataset/labels/train",
    "/home/elicer/try1/dataset/labels/val"
]

# 병합 대상 경로
dst_img_dir = "/home/elicer/try1/dataset/images"
dst_lbl_dir = "/home/elicer/try1/dataset/labels"

# 디렉토리 생성
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_lbl_dir, exist_ok=True)

# 이미지 파일 병합
for d in src_img_dirs:
    for file in glob(os.path.join(d, "*.png")):
        shutil.copy(file, dst_img_dir)

# 라벨 파일 병합
for d in src_lbl_dirs:
    for file in glob(os.path.join(d, "*.txt")):
        shutil.copy(file, dst_lbl_dir)

print("✅ 병합 완료: 모든 이미지 및 라벨을 dataset 디렉토리로 복사했습니다.")
