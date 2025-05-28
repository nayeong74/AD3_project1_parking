import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import confusion_matrix


# 1. 클래스 정의
id2label = {0: "Driving Area", 1: "Parking Area", 2: "No Parking Area"}
label2id = {v: k for k, v in id2label.items()}

# 2. Feature Extractor 로드
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)

# 3. PyTorch Dataset 클래스
class SegFormerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.images[idx])[0] + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        # feature_extractor는 이미지만 전처리
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs["labels"] = torch.tensor(mask, dtype=torch.long)

        return {k: v.squeeze() for k, v in inputs.items()}

# 4. 데이터셋 준비
train_dataset = SegFormerDataset(
    image_dir="/home/elicer/jny/dataset/images/train",
    mask_dir="/home/elicer/jny/dataset/masks/train",
    feature_extractor=feature_extractor
)

val_dataset = SegFormerDataset(
    image_dir="/home/elicer/jny/dataset/images/val",
    mask_dir="/home/elicer/jny/dataset/masks/val",
    feature_extractor=feature_extractor
)

# 5. 모델 로드
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.config.ignore_index = 255  # 무시 인덱스 설정

# 6. 성능 지표
import torch.nn.functional as F

def compute_metrics(p):
    id2label = {0: "Driving Area", 1: "Parking Area", 2: "No Parking Area"}
    num_classes = len(id2label)
    ignore_index = 255

    # 1. 라벨 처리
    labels = np.stack(p.label_ids)  # (N, H, W)
    label_shape = labels.shape[-2:]  # (H, W)

    # 2. 예측 처리
    # p.predictions: (N, C, h, w) → torch.Tensor로 변환 후 interpolate
    preds = torch.tensor(p.predictions)
    preds = F.interpolate(preds, size=label_shape, mode="bilinear", align_corners=False)
    preds = preds.argmax(dim=1).cpu().numpy()  # (N, H, W)

    # 3. flatten
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    # 4. 255 무시
    valid_mask = labels_flat != ignore_index
    preds_valid = preds_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]

    # 5. confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels_valid, preds_valid, labels=list(range(num_classes)))

    # 6. IoU
    ious = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        denom = tp + fn + fp
        iou = tp / denom if denom != 0 else 0.0
        ious.append(iou)

    miou = np.mean(ious)
    acc = np.trace(cm) / np.sum(cm)

    return {
        "mIoU": round(miou, 4),
        "pixel_accuracy": round(acc, 4),
        **{f"{id2label[i]}_IoU": round(ious[i], 4) for i in range(num_classes)}
    }




# 7. 학습 설정
training_args = TrainingArguments(
    output_dir="/home/elicer/jny/result/segformer-b0",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=2,
    fp16=True,  
    remove_unused_columns=True,  
    report_to="none"
)
# 8. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 9. 학습 시작
trainer.train()
