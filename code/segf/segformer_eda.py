import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# 경로 설정
img_path = "/home/elicer/jny/video/20221130_141642_60.png"
model_path = "/home/elicer/jny/result/segformer-b2/finetuned/checkpoint-6175"

# 동일한 Feature Extractor 인스턴스 (학습 시 설정과 일치)
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)

# 모델 로드
id2label = {0: "Background", 1: "Driving Area", 2: "Parking Area"}
label2id = {v: k for k, v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    model_path,
    id2label=id2label,
    label2id=label2id,
)
model.config.ignore_index = 255
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 불러오기
image = Image.open(img_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, num_labels, H, W)
    preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # (H, W)

# 마스크를 원본 이미지 크기로 리사이즈
original_size = image.size  # (W, H)
preds_resized = cv2.resize(preds.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

# 색상 매핑
class_colors = {
    0: (160, 160, 160),  # Background
    1: (0, 255, 0),      # Driving Area
    2: (0, 0, 255)       # Parking Area
}
image_np = np.array(image)
vis_mask = np.zeros_like(image_np)

for cls, color in class_colors.items():
    for c in range(3):
        vis_mask[:, :, c][preds_resized == cls] = color[c]

# 반투명 오버레이
overlay = cv2.addWeighted(image_np, 0.6, vis_mask, 0.4, 0)

# 저장
cv2.imwrite("segformer_prediction_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("✅ 저장 완료: segformer_prediction_overlay.png")
