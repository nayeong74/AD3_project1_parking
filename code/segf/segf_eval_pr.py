import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# ===================== #
# 1. 설정
# ===================== #
image_dir = "/home/elicer/jny/dataset/images/val"
mask_dir = "/home/elicer/jny/dataset/masks/val"
checkpoint_path = "/home/elicer/jny/result/checkpoint-2300"
output_plot_path = "/home/elicer/jny/result/mask_pr_curve.png"

id2label = {0: "Driving Area", 1: "Parking Area", 2: "No Parking Area"}
num_classes = len(id2label)
ignore_index = 255

# 모델 및 feature_extractor 로드
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)
model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
model.eval().cuda()

# ===================== #
# 2. 예측 및 정답 수집
# ===================== #
all_labels = []
all_preds = []

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

for fname in tqdm(image_files, desc="Evaluating"):
    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

    image = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path), dtype=np.uint8)

    # 입력 처리
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, num_classes, h, w]

        # GT 마스크 해상도에 맞춰 보간
        logits = F.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()

    # 🔥 255 (ignore_index) 마스크 무시
    valid_mask = mask != ignore_index
    mask = mask[valid_mask]
    pred = pred[valid_mask]

    all_labels.append(mask.flatten())
    all_preds.append(pred.flatten())

# ===================== #
# 3. PR Curve 계산 및 시각화
# ===================== #
all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

plt.figure(figsize=(10, 6))

for cls in range(num_classes):
    gt = (all_labels == cls).astype(np.uint8)
    pred = (all_preds == cls).astype(np.uint8)

    if np.sum(gt) == 0:
        print(f"[주의] 클래스 '{id2label[cls]}'에 대한 GT가 없습니다. 스킵합니다.")
        continue

    precision, recall, _ = precision_recall_curve(gt, pred)
    ap = average_precision_score(gt, pred)
    plt.plot(recall, precision, label=f"{id2label[cls]} (AP={ap:.2f})")

# ===================== #
# 4. 결과 저장
# ===================== #
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("SegFormer - Mask PR Curve)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot_path)
print(f"✔️ PR 그래프 저장 완료: {output_plot_path}")
