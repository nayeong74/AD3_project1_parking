from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ✅ 1. 마스크 경로 설정
mask_path = "/home/elicer/jny/dataset/train/masks/20220822_140059_10.png"
output_path = "/home/elicer/jny/dataset/visualized_mask.png" 

# ✅ 2. 마스크 로드 및 변환
mask = np.array(Image.open(mask_path), dtype=np.uint8)

# ✅ 3. 클래스별 색상 정의
color_map = {
    0: (255, 0, 0),     # 클래스 0: 빨강
    1: (0, 255, 0),     # 클래스 1: 초록
    2: (0, 0, 0),       # 클래스 2: 검정
   # 255: (160, 160, 160)  # 무시영역 (선택)
}

# ✅ 4. 시각화용 RGB 마스크 생성
color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
for cls_id, color in color_map.items():
    color_mask[mask == cls_id] = color

# ✅ 5. 이미지 저장
output_image = Image.fromarray(color_mask)
output_image.save(output_path)

print(f"✅ 시각화 이미지가 저장되었습니다: {output_path}")
print(f"🎯 사용된 클래스 ID: {np.unique(mask)}")
