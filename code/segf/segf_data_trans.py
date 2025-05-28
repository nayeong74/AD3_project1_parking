import os
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

# 변환 대상 디렉토리
splits = ["train", "val"]
base_image_dir = "/home/elicer/jny/dataset/images"
base_label_dir = "/home/elicer/jny/dataset/labels"
base_mask_dir = "/home/elicer/jny/dataset/masks"

# 에러 로그 저장
errors = []

for split in splits:
    image_dir = os.path.join(base_image_dir, split)
    label_dir = os.path.join(base_label_dir, split)
    mask_dir = os.path.join(base_mask_dir, split)
    os.makedirs(mask_dir, exist_ok=True)

    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith((".jpg", ".png")):
            continue

        try:
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("이미지를 열 수 없습니다.")

            h, w = image.shape[:2]
            # 🎯 기본값 = 255 (SegFormer에서 무시할 클래스)
            mask = np.full((h, w), fill_value=255, dtype=np.uint8)

            label_file = Path(img_file).with_suffix(".txt").name
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 7 or len(parts) % 2 == 0:
                            raise ValueError(f"[Line {line_num}] 포맷 오류: '{line.strip()}'")

                        class_id = int(parts[0])
                        coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                        abs_coords = np.round(coords * [w, h]).astype(np.int32)

                        cv2.fillPoly(mask, [abs_coords], color=class_id)

            out_path = os.path.join(mask_dir, Path(img_file).with_suffix(".png").name)
            cv2.imwrite(out_path, mask)

        except Exception as e:
            errors.append((split, img_file, str(e)))

# ✅ 에러 로그 출력
if errors:
    df = pd.DataFrame(errors, columns=["데이터셋", "이미지 파일", "오류 메시지"])
    print("❗ 변환 중 오류 발생:")
    print(df.to_string(index=False))
else:
    print("✅ 모든 마스크가 성공적으로 생성되었습니다.")
