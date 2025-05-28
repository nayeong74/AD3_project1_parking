from PIL import Image
import numpy as np

# 마스크 경로
mask_path = "/home/elicer/dataset/masks/train/20220822_135957_00.png"

# 마스크 로드 및 클래스 ID 확인
mask = np.array(Image.open(mask_path), dtype=np.uint8)
unique_ids = np.unique(mask)
print(f"🎯 이 마스크의 클래스 ID 목록: {unique_ids}")
