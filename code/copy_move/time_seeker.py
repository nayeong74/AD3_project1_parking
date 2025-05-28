import torch
import time
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

# === 설정 ===
checkpoint_path = "/home/elicer/jny/result/segformer-b0/checkpoint-2300"
image_path = "/home/elicer/jny/dataset/images/val/20220823_141224_40.png"  # 예시 이미지 경로

# === 모델 로드 ===
model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).cuda().eval()
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=False)

# === 파라미터 수
num_params = sum(p.numel() for p in model.parameters())
print(f"🔢 Total Parameters: {num_params:,}")

# === 입력 준비
image = Image.open(image_path).convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

# === GPU 메모리 초기화
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# === 추론 시간 측정
start = time.time()
with torch.no_grad():
    outputs = model(**inputs)
end = time.time()
elapsed = end - start

# === 메모리 사용량
max_mem = torch.cuda.max_memory_allocated() / 1024**2

# === 출력
print(f"⏱️ Inference Time: {elapsed:.4f} sec")
print(f"📦 Peak GPU Memory Usage: {max_mem:.2f} MB")
