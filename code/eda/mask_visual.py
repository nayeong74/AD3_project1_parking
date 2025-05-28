import cv2
import numpy as np

# 마스크 로드
mask_path = "/home/elicer/jny/dataset/masks/train/20220823_141302_10.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# 컬러맵 입혀서 저장
colored = cv2.applyColorMap((mask * 80).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite("/home/elicer/jny/test_mask_colored.png", colored)
print("✅ 저장 완료: test_mask_colored.png")
