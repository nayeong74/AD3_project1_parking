import os

# 라벨 없는 이미지에 대해 빈 txt 생성 (YOLOv8의 일반화 고려)
for split in ['train', 'val']:
    img_dir = f"/home/elicer/try1/dataset/images/{split}"
    lbl_dir = f"/home/elicer/try1/dataset/labels/{split}"
    for fname in os.listdir(img_dir):
        lbl_path = os.path.join(lbl_dir, fname.replace('.png', '.txt'))
        if not os.path.exists(lbl_path):
            open(lbl_path, 'w').close()