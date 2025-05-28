import os
import shutil
from pathlib import Path
from random import sample

def copy_fraction_images_and_labels(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, fraction=0.25):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.png'))])
    selected = sample(image_files, max(1, int(len(image_files) * fraction)))

    for img_file in selected:
        name = Path(img_file).stem
        label_file = f"{name}.png"

        shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

        src_label_path = os.path.join(src_lbl_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, os.path.join(dst_lbl_dir, label_file))

# 📁 원본 경로
SRC_IMG_BASE = "/home/elicer/dataset/images"
SRC_LBL_BASE = "/home/elicer/dataset/masks"

# 📁 복사 대상 경로
DST_IMG_BASE = "/home/elicer/jny/dataset/images"
DST_LBL_BASE = "/home/elicer/jny/dataset/masks"

# train/val 각각 복사
copy_fraction_images_and_labels(
    src_img_dir=os.path.join(SRC_IMG_BASE, "train"),
    src_lbl_dir=os.path.join(SRC_LBL_BASE, "train"),
    dst_img_dir=os.path.join(DST_IMG_BASE, "train"),
    dst_lbl_dir=os.path.join(DST_LBL_BASE, "train"),
    fraction=0.2
)

copy_fraction_images_and_labels(
    src_img_dir=os.path.join(SRC_IMG_BASE, "val"),
    src_lbl_dir=os.path.join(SRC_LBL_BASE, "val"),
    dst_img_dir=os.path.join(DST_IMG_BASE, "val"),
    dst_lbl_dir=os.path.join(DST_LBL_BASE, "val"),
    fraction=0.2
)
