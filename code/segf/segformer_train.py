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
        mask = np.array(Image.open(mask_path), dtype=np.uint8)

        # feature_extractor 이미지 전처리
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
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.config.ignore_index = 255  # 배경 무시 인덱스 설정

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="/home/elicer/jny/result",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    fp16=True,  
    remove_unused_columns=True,  
    report_to="none"
)
# 7. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 8. 학습 시작!
trainer.train()
