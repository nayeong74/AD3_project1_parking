import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# === 설정 ===
image_dir = "/home/elicer/dataset/images/val"
mask_dir = "/home/elicer/dataset/masks/val"
checkpoint_path = "/home/elicer/jny/result/segformer-b2/finetuned/checkpoint-6175"
id2label = {0: "Background", 1: "Driving Area", 2: "Parking Area"}
num_classes = len(id2label)
ignore_index = 255

# === 모델 불러오기 ===
model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)
model.eval().cuda()

# === Confusion Matrix 누적용 ===
conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
# 필요 시 이미지 수 제한:
# image_files = image_files[:100]

for fname in tqdm(image_files, desc="Evaluating"):
    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

    image = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path), dtype=np.uint8)

    # 입력 변환
    inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=mask.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # 유효한 픽셀만
    valid_mask = mask != ignore_index
    true = mask[valid_mask].flatten()
    pred = pred[valid_mask].flatten()

    # confusion matrix 누적
    cm = confusion_matrix(true, pred, labels=list(range(num_classes)))
    conf_matrix += cm.astype(np.uint64)

# === IoU 계산 ===
per_class_iou = []
for i in range(num_classes):
    tp = conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - tp
    fp = conf_matrix[:, i].sum() - tp
    denom = tp + fp + fn
    iou = tp / denom if denom != 0 else 0.0
    per_class_iou.append(iou)

mean_iou = np.mean(per_class_iou)
pixel_acc = np.trace(conf_matrix) / np.sum(conf_matrix)

# === 출력 ===
print("\n📊 SegFormer 성능 지표 (IoU 기반):")
for i, iou in enumerate(per_class_iou):
    print(f"- {id2label[i]} IoU: {iou:.4f}")
print(f"\n✅ Mean IoU: {mean_iou:.4f}")
print(f"✅ Pixel Accuracy: {pixel_acc:.4f}")
